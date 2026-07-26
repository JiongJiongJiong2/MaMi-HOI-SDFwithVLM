"""Print a real processed sample for manual 20/21/22/23 joint verification."""

import argparse
import csv
import json
from pathlib import Path

import joblib
import numpy as np


JOINT_SOURCES = {
    20: "SMPL-H base joint 20 (expected left wrist)",
    21: "SMPL-H base joint 21 (expected right wrist)",
    22: "SMPL-H Jtr[28] appended hand point (expected left palm proxy)",
    23: "SMPL-H Jtr[43] appended hand point (expected right palm proxy)",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output_folder", required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--sequence_name", default="")
    parser.add_argument("--window", type=int, default=120)
    return parser.parse_args()


def main():
    args = parse_args()
    data_root = Path(args.data_root_folder)
    processed_path = data_root / (
        f"cano_{args.split}_diffusion_manip_window_"
        f"{args.window}_joints24.p"
    )
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing {processed_path}. Run the normal MaMi-HOI dataset "
            "preprocessing once before this audit."
        )
    window_data = joblib.load(processed_path)
    candidates = [
        item
        for item in window_data.values()
        if not args.sequence_name or item["seq_name"] == args.sequence_name
    ]
    if not candidates:
        raise ValueError(f"No sample found for sequence {args.sequence_name!r}")
    sample = candidates[0]
    motion = np.asarray(sample["motion"])
    joints = motion[:, : 24 * 3].reshape(-1, 24, 3)
    if not np.isfinite(joints).all():
        raise ValueError("Selected sample contains NaN or Inf joints")

    output_folder = Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    frame_ids = sorted({0, len(joints) // 2, len(joints) - 1})
    csv_path = output_folder / "joint_ordering_real_sample.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("sequence", "frame", "joint_index", "source", "x", "y", "z"),
        )
        writer.writeheader()
        for frame in frame_ids:
            for joint_index, source in JOINT_SOURCES.items():
                xyz = joints[frame, joint_index]
                writer.writerow(
                    {
                        "sequence": sample["seq_name"],
                        "frame": frame,
                        "joint_index": joint_index,
                        "source": source,
                        "x": float(xyz[0]),
                        "y": float(xyz[1]),
                        "z": float(xyz[2]),
                    }
                )

    summary = {
        "sequence_name": sample["seq_name"],
        "split": args.split,
        "start_t_idx": int(sample["start_t_idx"]),
        "end_t_idx": int(sample["end_t_idx"]),
        "frames_written": frame_ids,
        "joint_sources": JOINT_SOURCES,
        "mean_distance_20_to_22_m": float(
            np.linalg.norm(joints[:, 20] - joints[:, 22], axis=-1).mean()
        ),
        "mean_distance_21_to_23_m": float(
            np.linalg.norm(joints[:, 21] - joints[:, 23], axis=-1).mean()
        ),
        "csv": str(csv_path),
        "manual_confirmation": [
            "20 and 22 must lie on the same anatomical left arm/hand.",
            "21 and 23 must lie on the same anatomical right arm/hand.",
            "22/23 are middle-finger-derived hand points; confirm they are acceptable palm proxies, not literal palm centers.",
            "Check that no left/right swap is visible across the three exported frames.",
        ],
    }
    summary_path = output_folder / "joint_ordering_summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
