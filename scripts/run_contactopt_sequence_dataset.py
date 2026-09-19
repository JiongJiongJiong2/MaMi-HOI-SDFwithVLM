#!/usr/bin/env python3
"""Run ContactOpt over a frozen sequence-disjoint window dataset."""

import argparse
import json
import os
import subprocess
from pathlib import Path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-json", type=Path, required=True)
    parser.add_argument("--case-script", type=Path, required=True)
    parser.add_argument("--python", type=Path, required=True)
    parser.add_argument("--sequence-db", type=Path, required=True)
    parser.add_argument("--workdir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--geometry-dir", type=Path, required=True)
    parser.add_argument(
        "--split",
        nargs="+",
        choices=("train", "dev", "test"),
        default=("train", "dev"),
    )
    parser.add_argument("--sequence", nargs="*", default=[])
    parser.add_argument("--max-frames-per-chunk", type=int, default=16)
    parser.add_argument("--max-chunks", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--n-iter", type=int, default=250)
    parser.add_argument("--contact-p10-threshold", type=float, default=0.02)
    parser.add_argument("--min-eligible-frames", type=int, default=3)
    parser.add_argument("--tag-prefix", default="sequence_dataset_v1")
    return parser.parse_args()


def build_chunks(windows, max_frames_per_chunk):
    if max_frames_per_chunk < 1:
        raise ValueError("max_frames_per_chunk must be positive")
    chunks = []
    current = []
    current_frames = set()
    for window in windows:
        frames = {int(frame) for frame in window["frames"]}
        if current and len(current_frames | frames) > max_frames_per_chunk:
            chunks.append(current)
            current = []
            current_frames = set()
        current.append(window)
        current_frames.update(frames)
    if current:
        chunks.append(current)
    return chunks


def group_windows(manifest, splits, sequences, allow_test):
    selected_splits = set(splits)
    if "test" in selected_splits and not allow_test:
        raise ValueError("test split requires --allow-test")
    selected_sequences = set(sequences)
    grouped = {}
    for window in manifest["windows"]:
        split = window["split"]
        sequence = window["sequence"]
        if split not in selected_splits:
            continue
        if selected_sequences and sequence not in selected_sequences:
            continue
        grouped.setdefault(sequence, []).append(window)
    return grouped


def complete_existing(output_json, geometry_npz):
    return output_json.is_file() and geometry_npz.is_file()


def summarize_case(result, chunk):
    windows = chunk["windows"]
    summary = {
        "chunk_id": chunk["chunk_id"],
        "sequence": chunk["sequence"],
        "split": chunk["split"],
        "object_name": chunk["object_name"],
        "candidate_npz": chunk["candidate_npz"],
        "frames": chunk["frames"],
        "window_count": len(windows),
        "windows": [window["frames"] for window in windows],
        "output_json": str(chunk["output_json"]),
        "geometry_npz": str(chunk["geometry_npz"]),
    }
    if result is None:
        summary["status"] = "skipped_existing"
        return summary
    summary.update(
        {
            "status": "completed",
            "gate_passed": bool(result.get("gate", {}).get("passed", False)),
            "eligible_frame_count": len(result.get("eligible_frames", [])),
            "contact_improved_frames": result.get("contact_improved_frames"),
            "distance_improved_frames": result.get("distance_improved_frames"),
            "mean_hand_contact_relative_change": result.get(
                "mean_hand_contact_relative_change"
            ),
            "mean_distance_relative_change": result.get(
                "mean_distance_relative_change"
            ),
            "max_wrist_drift_m": result.get("max_wrist_drift_m"),
            "max_object_drift_m": result.get("max_object_drift_m"),
            "optimized_pkl": result.get("optimized_pkl"),
        }
    )
    return summary


def make_chunks(grouped, max_frames_per_chunk, tag_prefix):
    chunks = []
    for sequence in sorted(grouped):
        windows = sorted(
            grouped[sequence],
            key=lambda item: (item["frames"][0], item["frames"][-1]),
        )
        for index, window_chunk in enumerate(
            build_chunks(windows, max_frames_per_chunk)
        ):
            frames = sorted(
                {
                    int(frame)
                    for window in window_chunk
                    for frame in window["frames"]
                }
            )
            chunk_id = f"{sequence}_c{index:03d}"
            chunks.append(
                {
                    "chunk_id": chunk_id,
                    "tag": f"{tag_prefix}_{chunk_id}",
                    "sequence": sequence,
                    "split": window_chunk[0]["split"],
                    "object_name": window_chunk[0]["object_name"],
                    "candidate_npz": window_chunk[0]["path"],
                    "frames": frames,
                    "windows": window_chunk,
                }
            )
    return chunks


def write_summary(path, manifest, args, rows):
    completed = [row for row in rows if row["status"] == "completed"]
    summary = {
        "manifest_json": str(args.manifest_json),
        "manifest_sha256": manifest.get("manifest_sha256"),
        "splits": list(args.split),
        "sequence_filter": args.sequence,
        "max_frames_per_chunk": args.max_frames_per_chunk,
        "chunk_count": len(rows),
        "completed_chunk_count": len(completed),
        "skipped_chunk_count": len(rows) - len(completed),
        "sequence_count": len({row["sequence"] for row in rows}),
        "window_count": sum(row["window_count"] for row in rows),
        "frame_count": sum(len(row["frames"]) for row in rows),
        "passed_chunk_count": sum(
            bool(row.get("gate_passed")) for row in completed
        ),
        "rows": rows,
    }
    path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return summary


def main():
    args = parse_args()
    manifest = json.loads(
        args.manifest_json.read_text(encoding="utf-8")
    )
    if not manifest.get("valid_sequence_disjoint_dataset", False):
        raise ValueError("Dataset manifest is not marked valid")

    grouped = group_windows(
        manifest,
        args.split,
        args.sequence,
        args.allow_test,
    )
    chunks = make_chunks(
        grouped,
        args.max_frames_per_chunk,
        args.tag_prefix,
    )
    if args.max_chunks is not None:
        chunks = chunks[: args.max_chunks]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.geometry_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["OMP_NUM_THREADS"] = "8"
    env["MKL_NUM_THREADS"] = "8"
    env.pop("http_proxy", None)
    env.pop("https_proxy", None)

    rows = []
    summary_path = args.output_dir / "batch_summary.json"
    for index, chunk in enumerate(chunks):
        output_json = args.output_dir / f"{chunk['chunk_id']}.json"
        geometry_npz = args.geometry_dir / f"{chunk['chunk_id']}.npz"
        chunk["output_json"] = output_json
        chunk["geometry_npz"] = geometry_npz

        if args.resume and complete_existing(output_json, geometry_npz):
            result = None
        else:
            command = [
                str(args.python),
                str(args.case_script),
                "--candidate-npz",
                chunk["candidate_npz"],
                "--sequence-db",
                str(args.sequence_db),
                "--contact-p10-threshold",
                str(args.contact_p10_threshold),
                "--min-eligible-frames",
                str(args.min_eligible_frames),
                "--n-iter",
                str(args.n_iter),
                "--output-tag",
                chunk["tag"],
                "--output-json",
                str(output_json),
                "--save-geometry-npz",
                str(geometry_npz),
                "--frames",
                *[str(frame) for frame in chunk["frames"]],
            ]
            print(
                f"[{index + 1}/{len(chunks)}] "
                f"{chunk['chunk_id']} frames={len(chunk['frames'])}",
                flush=True,
            )
            subprocess.run(
                command,
                cwd=args.workdir,
                env=env,
                check=True,
            )
            result = json.loads(output_json.read_text(encoding="utf-8"))

        rows.append(summarize_case(result, chunk))
        write_summary(summary_path, manifest, args, rows)

    summary = write_summary(summary_path, manifest, args, rows)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
