#!/usr/bin/env python3
"""Render local 3-D ContactOpt input/refined comparisons as GIFs."""

import argparse
import io
import pickle
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy.spatial import cKDTree


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--optimized-pkl", type=Path, nargs="+", required=True)
    parser.add_argument("--label", action="append", required=True)
    parser.add_argument("--output-gif", type=Path, required=True)
    parser.add_argument("--duration-ms", type=int, default=550)
    parser.add_argument("--object-radius-m", type=float, default=0.22)
    parser.add_argument("--max-object-points", type=int, default=4500)
    parser.add_argument("--dpi", type=int, default=100)
    return parser.parse_args()


def local_object_points(object_vertices, center, radius, max_points):
    distances = np.linalg.norm(object_vertices - center, axis=1)
    selected = object_vertices[distances <= radius]
    if len(selected) > max_points:
        stride = int(np.ceil(len(selected) / max_points))
        selected = selected[::stride]
    return selected


def hand_center(run):
    points = np.concatenate(
        [run["in_ho"].hand_verts, run["out_ho"].hand_verts], axis=0
    )
    return np.median(points, axis=0)


def prepare_case(pkl_path, label, radius, max_object_points):
    with pkl_path.open("rb") as handle:
        runs = pickle.load(handle)
    frames = []
    for run in runs:
        input_hand = run["in_ho"]
        refined_hand = run["out_ho"]
        center = hand_center(run)
        object_points = local_object_points(
            input_hand.obj_verts, center, radius, max_object_points
        )
        input_distance = cKDTree(input_hand.obj_verts).query(
            input_hand.hand_verts
        )[0]
        refined_distance = cKDTree(refined_hand.obj_verts).query(
            refined_hand.hand_verts
        )[0]
        frames.append(
            {
                "object": object_points,
                "input": input_hand.hand_verts,
                "refined": refined_hand.hand_verts,
                "input_distance_mean": float(input_distance.mean()),
                "refined_distance_mean": float(refined_distance.mean()),
                "center": center,
            }
        )

    return {
        "label": label,
        "frames": frames,
    }


def render_frame(cases, frame_index, total_frames):
    figure, axes = plt.subplots(
        1,
        len(cases),
        figsize=(4.6 * len(cases), 4.2),
        subplot_kw={"projection": "3d"},
    )
    if len(cases) == 1:
        axes = [axes]
    figure.patch.set_facecolor("#f7f7f5")

    for axis, case in zip(axes, cases):
        frame = case["frames"][frame_index]
        to_local = lambda points: (points - frame["center"]) * 1000.0

        object_points = to_local(frame["object"])
        input_points = to_local(frame["input"])
        refined_points = to_local(frame["refined"])

        axis.scatter(
            object_points[:, 0],
            object_points[:, 1],
            object_points[:, 2],
            s=1.8,
            c="#9ca3af",
            alpha=0.45,
            depthshade=False,
        )
        axis.scatter(
            input_points[:, 0],
            input_points[:, 1],
            input_points[:, 2],
            s=4.0,
            c="#f97316",
            alpha=0.72,
            depthshade=False,
        )
        axis.scatter(
            refined_points[:, 0],
            refined_points[:, 1],
            refined_points[:, 2],
            s=4.0,
            c="#2563eb",
            alpha=0.82,
            depthshade=False,
        )
        axis.set_xlim(-110, 110)
        axis.set_ylim(-110, 110)
        axis.set_zlim(-110, 110)
        axis.set_box_aspect((1, 1, 1))
        axis.view_init(elev=22, azim=-58)
        axis.set_axis_off()
        axis.set_title(
            f"{case['label']} | frame {frame_index + 1}/{total_frames}",
            fontsize=11,
        )
        axis.text2D(
            0.02,
            0.02,
            "mean nearest distance: "
            f"{frame['input_distance_mean'] * 1000:.1f} mm -> "
            f"{frame['refined_distance_mean'] * 1000:.1f} mm",
            transform=axis.transAxes,
            fontsize=9,
            color="#111827",
        )

    figure.text(
        0.5,
        0.015,
        "orange: MaMi input | blue: ContactOpt refined | grey: local object surface",
        ha="center",
        fontsize=9,
        color="#374151",
    )
    figure.subplots_adjust(left=0.01, right=0.99, top=0.91, bottom=0.08)
    buffer = io.BytesIO()
    figure.savefig(buffer, format="png", dpi=100, facecolor=figure.get_facecolor())
    plt.close(figure)
    buffer.seek(0)
    return Image.open(buffer).convert("RGB")


def main():
    args = parse_args()
    if len(args.optimized_pkl) != len(args.label):
        raise ValueError("--optimized-pkl and --label counts differ")

    cases = [
        prepare_case(
            pkl_path,
            label,
            args.object_radius_m,
            args.max_object_points,
        )
        for pkl_path, label in zip(args.optimized_pkl, args.label)
    ]
    total_frames = min(len(case["frames"]) for case in cases)

    images = [
        render_frame(cases, index, total_frames)
        for index in range(total_frames)
    ]
    args.output_gif.parent.mkdir(parents=True, exist_ok=True)
    images[0].save(
        args.output_gif,
        save_all=True,
        append_images=images[1:],
        duration=args.duration_ms,
        loop=0,
        optimize=True,
        disposal=2,
    )
    print(args.output_gif)


if __name__ == "__main__":
    main()
