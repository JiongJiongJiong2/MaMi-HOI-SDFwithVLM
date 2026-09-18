#!/usr/bin/env python3
"""Run ContactOpt on real MaMi hand-object outputs with frozen wrist/object.

The script converts the saved SMPLX right-hand mesh into an aligned MANO hand,
builds ContactOpt's expected sample dictionary, runs batched refinement, and
writes a JSON summary with the pre-registered Phase 7 gate.
"""

import argparse
import json
import os
import pickle
import sys
from pathlib import Path
from types import SimpleNamespace

import joblib
import numpy as np
import open3d as o3d
import torch
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--sequence-db", type=Path, required=True)
    parser.add_argument(
        "--contactopt-root",
        type=Path,
        default=Path("/root/autodl-tmp/external/ContactOpt"),
    )
    parser.add_argument(
        "--body-model-module-root",
        type=Path,
        default=Path("/root/autodl-tmp/mami-wm-c537e84/utils"),
    )
    parser.add_argument(
        "--body-model-root",
        type=Path,
        default=Path(
            "/root/autodl-tmp/mamihoi/data/processed_data/smpl_all_models"
        ),
    )
    parser.add_argument(
        "--right-hand-vids",
        type=Path,
        default=Path(
            "/root/autodl-tmp/mami-wm-c537e84/data/part_vert_ids/"
            "right_hand_vids.npy"
        ),
    )
    parser.add_argument("--num-frames", type=int, default=10)
    parser.add_argument("--min-frame-separation", type=int, default=5)
    parser.add_argument("--frames", type=int, nargs="*")
    parser.add_argument("--n-iter", type=int, default=250)
    parser.add_argument("--output-tag", default="mami_phase7")
    parser.add_argument("--output-json", type=Path, required=True)
    return parser.parse_args()


def load_sequence_record(sequence_db, sequence_name):
    records = joblib.load(sequence_db)
    for record in records.values():
        if str(record["seq_name"]) == sequence_name:
            return record
    raise KeyError(f"Sequence {sequence_name!r} is absent from {sequence_db}")


def procrustes(source, target):
    source_centroid = source.mean(axis=0)
    target_centroid = target.mean(axis=0)
    source_centered = source - source_centroid
    target_centered = target - target_centroid
    u, _, vt = np.linalg.svd(source_centered.T @ target_centered)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1] *= -1
        rotation = vt.T @ u.T
    translation = target_centroid - rotation @ source_centroid
    return rotation, translation


def transform_points(points, rotation, translation):
    return points @ rotation.T + translation


def make_body_model(body_model_module_root, body_model_root, gender):
    sys.path.insert(0, str(body_model_module_root))
    from human_body_prior.body_model.body_model import BodyModel

    model_path = (
        body_model_root / "smplx" / f"SMPLX_{gender.upper()}.npz"
    )
    return BodyModel(
        bm_fname=str(model_path),
        num_betas=16,
        num_expressions=None,
        dtype=torch.float64,
    ).double().cuda()


def body_hand_geometry(body_model, betas, right_hand_vids):
    zeros3 = torch.zeros(1, 3, device="cuda", dtype=torch.float64)
    zeros63 = torch.zeros(1, 63, device="cuda", dtype=torch.float64)
    zeros90 = torch.zeros(1, 90, device="cuda", dtype=torch.float64)
    with torch.no_grad():
        output = body_model(
            root_orient=zeros3,
            pose_body=zeros63,
            pose_hand=zeros90,
            betas=torch.tensor(
                betas, device="cuda", dtype=torch.float64
            ).reshape(1, -1),
            trans=zeros3,
        )
    vertices = output.v[0].detach().cpu().numpy()
    joints = output.Jtr[0].detach().cpu().numpy()
    return vertices[right_hand_vids], joints


def mano_neutral_geometry(contactopt_root, beta10):
    sys.path.insert(0, str(contactopt_root))
    from manopth.manolayer import ManoLayer

    mano = ManoLayer(
        mano_root=str(contactopt_root / "mano" / "models"),
        use_pca=True,
        ncomps=15,
        side="right",
        flat_hand_mean=False,
    )
    vertices, joints = mano(
        torch.zeros(1, 18),
        torch.tensor(beta10[None], dtype=torch.float32),
    )
    return (
        vertices[0].detach().cpu().numpy().astype(np.float64) / 1000.0,
        joints[0].detach().cpu().numpy().astype(np.float64) / 1000.0,
    )


def align_mano_to_saved_hand(
    contactopt_root,
    canonical_hand,
    canonical_joints,
    saved_hand,
    beta10,
):
    mano_hand, mano_joints = mano_neutral_geometry(contactopt_root, beta10)

    rotation_ct, translation_ct = procrustes(canonical_hand, saved_hand)
    canonical_residual = np.linalg.norm(
        transform_points(
            canonical_hand, rotation_ct, translation_ct
        )
        - saved_hand,
        axis=1,
    )

    source_axis = mano_joints[6] - mano_joints[0]
    target_axis = canonical_joints[43] - canonical_joints[21]
    source_axis /= np.linalg.norm(source_axis)
    target_axis /= np.linalg.norm(target_axis)
    rotation_mc = Rotation.align_vectors(
        target_axis[None], source_axis[None]
    )[0].as_matrix()
    translation_mc = canonical_joints[21] - rotation_mc @ mano_joints[0]

    mano_to_canonical = np.eye(4)
    mano_to_canonical[:3, :3] = rotation_mc
    mano_to_canonical[:3, 3] = translation_mc

    registration = o3d.pipelines.registration.registration_icp(
        o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(mano_hand)
        ),
        o3d.geometry.PointCloud(
            o3d.utility.Vector3dVector(canonical_hand)
        ),
        0.05,
        mano_to_canonical,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(),
        o3d.pipelines.registration.ICPConvergenceCriteria(
            max_iteration=200
        ),
    )
    mano_to_canonical = registration.transformation

    canonical_to_saved = np.eye(4)
    canonical_to_saved[:3, :3] = rotation_ct
    canonical_to_saved[:3, 3] = translation_ct
    mano_to_saved = canonical_to_saved @ mano_to_canonical

    transformed_mano = transform_points(
        mano_hand, mano_to_saved[:3, :3], mano_to_saved[:3, 3]
    )
    tree = cKDTree(saved_hand[::4])
    mano_to_saved_distance = tree.query(transformed_mano)[0]

    return mano_to_saved, {
        "canonical_residual_mean_m": float(canonical_residual.mean()),
        "canonical_residual_max_m": float(canonical_residual.max()),
        "mano_to_saved_mean_m": float(mano_to_saved_distance.mean()),
        "mano_to_saved_median_m": float(np.median(mano_to_saved_distance)),
        "mano_to_saved_max_m": float(mano_to_saved_distance.max()),
        "icp_fitness": float(registration.fitness),
        "icp_inlier_rmse_m": float(registration.inlier_rmse),
    }


def select_frames(candidate, requested_frames, num_frames, min_separation):
    hand = np.asarray(candidate["pred_right_hand_verts"])
    obj = np.asarray(candidate["pred_object_verts"])
    if requested_frames:
        return sorted(requested_frames)

    distances = []
    for frame in range(len(hand)):
        nearest = cKDTree(obj[frame, ::50]).query(hand[frame])[0]
        distances.append((float(nearest.min()), frame))

    selected = []
    for _, frame in sorted(distances):
        if all(abs(frame - other) >= min_separation for other in selected):
            selected.append(frame)
        if len(selected) == num_frames:
            break
    if len(selected) != num_frames:
        raise ValueError(
            f"Could only select {len(selected)} frames with separation "
            f"{min_separation}"
        )
    return sorted(selected)


def build_samples(args, candidate, sequence_record, right_hand_vids):
    contactopt_root = args.contactopt_root
    body_model = make_body_model(
        args.body_model_module_root,
        args.body_model_root,
        str(sequence_record["gender"]),
    )
    betas = np.asarray(
        sequence_record["betas"], dtype=np.float64
    ).reshape(-1)
    canonical_hand, canonical_joints = body_hand_geometry(
        body_model, betas, right_hand_vids
    )

    frames = select_frames(
        candidate,
        args.frames,
        args.num_frames,
        args.min_frame_separation,
    )
    saved_right_hand = np.asarray(
        candidate["pred_right_hand_verts"], dtype=np.float64
    )
    object_vertices = np.asarray(
        candidate["pred_object_verts"], dtype=np.float32
    )
    object_faces = np.asarray(candidate["object_faces"], dtype=np.int64)

    from contactopt import util
    from contactopt.hand_object import HandObject

    samples = []
    alignment = []
    for frame in frames:
        transform, diagnostics = align_mano_to_saved_hand(
            contactopt_root,
            canonical_hand,
            canonical_joints,
            saved_right_hand[frame],
            betas[:10].astype(np.float32),
        )
        rotation = Rotation.from_matrix(transform[:3, :3])
        hand_pose = np.concatenate(
            [rotation.as_rotvec(), np.zeros(15, dtype=np.float64)]
        )

        hand_object = HandObject()
        hand_object.load_from_mano_params(
            hand_beta=betas[:10].astype(np.float32),
            hand_pose=hand_pose,
            hand_trans=transform[:3, 3].astype(np.float32),
            obj_faces=object_faces,
            obj_verts=object_vertices[frame],
        )
        ground_truth = HandObject()
        ground_truth.load_from_ho(hand_object)

        rng = np.random.default_rng(10000 + frame)
        sampled_idx = rng.choice(
            len(object_vertices[frame]),
            size=min(util.SAMPLE_VERTS_NUM, len(object_vertices[frame])),
            replace=False,
        )
        hand_features, object_features = (
            hand_object.generate_pointnet_features(sampled_idx)
        )
        samples.append(
            {
                "ho_aug": hand_object,
                "ho_gt": ground_truth,
                "obj_sampled_idx": sampled_idx,
                "hand_feats_aug": hand_features,
                "obj_feats_aug": object_features,
            }
        )
        alignment.append({"frame": frame, **diagnostics})

    return samples, frames, alignment


def make_contactopt_args(samples, output_tag, n_iter):
    return SimpleNamespace(
        batch_size=len(samples),
        split=output_tag,
        lr=0.01,
        n_iter=n_iter,
        partial=-1,
        w_cont_hand=2.5,
        sharpen_thresh=-1,
        ncomps=15,
        w_cont_asym=2.0,
        w_opt_trans=0.0,
        w_opt_rot=0.0,
        w_opt_pose=1.0,
        caps_rad=0.001,
        caps_hand=False,
        cont_method=0,
        caps_top=0.0005,
        caps_bot=-0.001,
        w_pen_cost=320.0,
        pen_it=0,
        w_obj_rot=0.0,
        rand_re=0,
        rand_re_trans=0.0,
        rand_re_rot=0.0,
        vis_method=1,
        vis=False,
        video=False,
        min_cont=1,
        test_dataset=samples,
    )


def evaluate_runs(optimized_path):
    with optimized_path.open("rb") as handle:
        runs = pickle.load(handle)

    rows = []
    for index, run in enumerate(runs):
        input_hand = run["in_ho"]
        refined_hand = run["out_ho"]
        input_hand.calc_dist_contact(hand=True, obj=True)
        refined_hand.calc_dist_contact(hand=True, obj=True)

        input_dist = cKDTree(input_hand.obj_verts).query(
            input_hand.hand_verts
        )[0]
        refined_dist = cKDTree(refined_hand.obj_verts).query(
            refined_hand.hand_verts
        )[0]
        wrist_drift = np.linalg.norm(
            refined_hand.hand_joints[0] - input_hand.hand_joints[0]
        )
        object_drift = np.linalg.norm(
            refined_hand.obj_verts - input_hand.obj_verts, axis=1
        ).max()
        input_hand_contact = input_hand.hand_contact.mean()
        refined_hand_contact = refined_hand.hand_contact.mean()
        input_object_contact = input_hand.obj_contact.mean()
        refined_object_contact = refined_hand.obj_contact.mean()

        if (
            not np.isfinite(input_hand_contact)
            or not np.isfinite(refined_hand_contact)
            or input_hand_contact <= 0
        ):
            hand_contact_change = None
        else:
            hand_contact_change = float(
                refined_hand_contact / input_hand_contact - 1.0
            )

        rows.append(
            {
                "sample_index": index,
                "input_distance_mean_m": float(input_dist.mean()),
                "refined_distance_mean_m": float(refined_dist.mean()),
                "distance_relative_change": float(
                    refined_dist.mean() / input_dist.mean() - 1.0
                ),
                "input_hand_contact_mean": float(
                    input_hand_contact
                )
                if np.isfinite(input_hand_contact)
                else None,
                "refined_hand_contact_mean": float(
                    refined_hand_contact
                )
                if np.isfinite(refined_hand_contact)
                else None,
                "hand_contact_relative_change": hand_contact_change,
                "input_object_contact_mean": float(
                    input_object_contact
                )
                if np.isfinite(input_object_contact)
                else None,
                "refined_object_contact_mean": float(
                    refined_object_contact
                )
                if np.isfinite(refined_object_contact)
                else None,
                "wrist_root_drift_m": float(wrist_drift),
                "object_vertex_drift_m": float(object_drift),
                "hand_vertex_motion_mean_m": float(
                    np.linalg.norm(
                        refined_hand.hand_verts - input_hand.hand_verts,
                        axis=1,
                    ).mean()
                ),
            }
        )

    return rows


def aggregate_rows(rows, alignment):
    contact_improvements = [
        row["hand_contact_relative_change"]
        for row in rows
        if row["hand_contact_relative_change"] is not None
    ]
    distance_changes = [row["distance_relative_change"] for row in rows]
    distance_improvements = [
        change < 0 for change in distance_changes
    ]
    return {
        "frames": len(rows),
        "alignment_canonical_mean_m": float(
            np.mean([item["canonical_residual_mean_m"] for item in alignment])
        ),
        "alignment_mano_mean_m": float(
            np.mean([item["mano_to_saved_mean_m"] for item in alignment])
        ),
        "contact_improved_frames": int(
            sum(change > 0 for change in contact_improvements)
        ),
        "valid_contact_frames": len(contact_improvements),
        "distance_improved_frames": int(sum(distance_improvements)),
        "mean_hand_contact_relative_change": float(
            np.mean(contact_improvements)
        )
        if contact_improvements
        else None,
        "mean_distance_relative_change": float(
            np.mean(distance_changes)
        ),
        "max_wrist_drift_m": float(
            max(row["wrist_root_drift_m"] for row in rows)
        ),
        "max_object_drift_m": float(
            max(row["object_vertex_drift_m"] for row in rows)
        ),
        "mean_hand_vertex_motion_m": float(
            np.mean([row["hand_vertex_motion_mean_m"] for row in rows])
        ),
    }


def main():
    args = parse_args()
    candidate_path = args.candidate_npz.resolve()
    sequence_db = args.sequence_db.resolve()
    output_json = args.output_json.resolve()
    contactopt_root = args.contactopt_root.resolve()
    right_hand_vids = np.load(args.right_hand_vids)

    candidate = np.load(candidate_path, allow_pickle=True)
    sequence_name = str(candidate["seq_name"])
    sequence_record = load_sequence_record(sequence_db, sequence_name)

    samples, frames, alignment = build_samples(
        args, candidate, sequence_record, right_hand_vids
    )

    sys.path.insert(0, str(contactopt_root))
    os.chdir(contactopt_root)
    from contactopt.run_contactopt import run_contactopt

    contactopt_args = make_contactopt_args(
        samples, args.output_tag, args.n_iter
    )
    run_contactopt(contactopt_args)

    optimized_path = (
        contactopt_root / "data" / f"optimized_{args.output_tag}.pkl"
    )
    rows = evaluate_runs(optimized_path)
    summary = aggregate_rows(rows, alignment)
    summary.update(
        {
            "candidate_npz": str(candidate_path),
            "sequence": sequence_name,
            "object_name": str(candidate["object_name"]),
            "frames": frames,
            "optimized_pkl": str(optimized_path),
            "rows": rows,
            "alignment": alignment,
        }
    )
    summary["gate"] = {
        "canonical_alignment": summary["alignment_canonical_mean_m"] <= 0.003,
        "mano_alignment": summary["alignment_mano_mean_m"] <= 0.015,
        "wrist_frozen": summary["max_wrist_drift_m"] <= 0.00001,
        "object_frozen": summary["max_object_drift_m"] <= 0.000001,
        "valid_contact_metric": summary["valid_contact_frames"] >= 7,
        "contact_improved": summary["contact_improved_frames"] >= 7,
        "distance_not_regressed": summary[
            "mean_distance_relative_change"
        ]
        <= 0.10,
    }
    summary["gate"]["passed"] = all(summary["gate"].values())

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summary, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
