#!/usr/bin/env python3
"""Run the frozen P0 training arms and aggregate their gate."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


ARMS = (
    ("listwise", "all"),
    ("listwise", "none"),
    ("absolute", "all"),
    ("quotient", "all"),
    ("oracle", "all"),
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seeds", nargs="+", type=int, default=(11, 23, 37))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument("--bootstrap", type=int, default=10000)
    return parser.parse_args()


def main():
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for objective, condition in ARMS:
        for seed in args.seeds:
            output = (
                args.output_dir
                / f"{objective}_{condition.replace(':', '_')}_seed{seed}"
            )
            if (output / "metrics.json").exists():
                print(f"SKIP {output}", flush=True)
                continue
            command = [
                sys.executable,
                str(root / "scripts" / "train_dwm_probe_rank.py"),
                "--dataset-dir",
                str(args.dataset_dir),
                "--output-dir",
                str(output),
                "--objective",
                objective,
                "--condition",
                condition,
                "--epochs",
                str(args.epochs),
                "--patience",
                str(args.patience),
                "--seed",
                str(seed),
                "--device",
                args.device,
            ]
            if args.allow_test:
                command.append("--allow-test")
            print("RUN", " ".join(command), flush=True)
            subprocess.run(command, check=True)
    if args.allow_test:
        subprocess.run([
            sys.executable,
            str(root / "scripts" / "analyze_dwm_probe_results.py"),
            "--dataset-dir",
            str(args.dataset_dir),
            "--results-dir",
            str(args.output_dir),
            "--output",
            str(args.output_dir / "p0_gate.json"),
            "--bootstrap",
            str(args.bootstrap),
        ], check=True)


if __name__ == "__main__":
    main()
