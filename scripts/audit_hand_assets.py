#!/usr/bin/env python3
"""Audit hand assets and MaMi processed data on a CPU-only host."""

from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import joblib
import numpy as np
import torch

from utils.human_body_prior.body_model.body_model import BodyModel


RAW_CANDIDATES = (
    "raw_behave",
    "behave-30fps-params",
    "GRAB",
    "grab",
    "arctic",
    "ARCTIC",
    "hot3d",
    "HOT3D",
    "EPIC-Contact",
    "epic_contact",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def inspect_pickle(path):
    with path.open("rb") as handle:
        data = pickle.load(handle, encoding="latin1")
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "keys": sorted(data.keys()),
        "v_template": list(data["v_template"].shape),
        "faces": list(data["f"].shape),
        "kintree": list(data["kintree_table"].shape),
        "j_regressor": list(data["J_regressor"].shape),
    }


def inspect_npz(path):
    with np.load(path, allow_pickle=True) as data:
        return {
            "path": str(path),
            "size": int(path.stat().st_size),
            "keys": sorted(data.files),
            "v_template": list(data["v_template"].shape),
            "faces": list(data["f"].shape),
            "kintree": list(data["kintree_table"].shape),
            "j_regressor": list(data["J_regressor"].shape),
        }


def inspect_processed_sequence(path):
    data = joblib.load(path)
    first = data[0]
    return {
        "path": str(path),
        "size": int(path.stat().st_size),
        "sequence_count": len(data),
        "keys": sorted(first.keys()),
        "has_pose_hand": "pose_hand" in first,
        "has_pose_body": "pose_body" in first,
    }


def audit_smplx_hand_effect(path):
    body_model = BodyModel(
        bm_fname=str(path),
        num_betas=10,
    ).cpu().eval()
    with torch.no_grad():
        pose_body = torch.zeros(1, 63)
        pose_hand = torch.zeros(1, 90)
        betas = torch.zeros(1, 10)
        neutral = body_model(
            pose_body=pose_body,
            pose_hand=pose_hand,
            betas=betas,
        )
        perturbed_hand = pose_hand.clone()
        perturbed_hand[0, 0] = 0.5
        perturbed = body_model(
            pose_body=pose_body,
            pose_hand=perturbed_hand,
            betas=betas,
        )
    delta = (neutral.v - perturbed.v).abs()
    return {
        "joint_count": int(neutral.Jtr.shape[1]),
        "vertex_count": int(neutral.v.shape[1]),
        "perturbation_max_vertex_delta": float(delta.max().item()),
        "perturbation_mean_vertex_delta": float(delta.mean().item()),
    }


def main():
    args = parse_args()
    data_root = Path(args.data_root).resolve()
    raw_candidates = [
        str(data_root / name)
        for name in RAW_CANDIDATES
        if (data_root / name).exists()
    ]
    result = {
        "data_root": str(data_root),
        "raw_hand_data_candidates": raw_candidates,
        "raw_hand_data_found": bool(raw_candidates),
        "processed_sequence": inspect_processed_sequence(
            data_root / "test_diffusion_manip_seq_joints24.p"
        ),
        "assets": {
            "mano_left": inspect_pickle(
                data_root
                / "smpl_all_models"
                / "mano"
                / "MANO_LEFT.pkl"
            ),
            "mano_right": inspect_pickle(
                data_root
                / "smpl_all_models"
                / "mano"
                / "MANO_RIGHT.pkl"
            ),
            "smplh_male": inspect_pickle(
                data_root
                / "smpl_all_models"
                / "smplh"
                / "SMPLH_male.pkl"
            ),
            "smplx_male": inspect_npz(
                data_root
                / "smpl_all_models"
                / "smplx"
                / "SMPLX_MALE.npz"
            ),
        },
        "smplx_hand_effect": audit_smplx_hand_effect(
            data_root
            / "smpl_all_models"
            / "smplx"
            / "SMPLX_MALE.npz"
        ),
    }
    result["gates"] = {
        "e3a_raw_finger_supervision": "pass"
        if result["raw_hand_data_found"]
        else "blocked",
        "e3b_hand_model_assets": "pass",
        "e3b_processed_cache_pose_hand": "pass"
        if result["processed_sequence"]["has_pose_hand"]
        else "blocked",
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

