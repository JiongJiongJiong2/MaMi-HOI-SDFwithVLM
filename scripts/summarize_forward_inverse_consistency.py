#!/usr/bin/env python3
"""Aggregate E2 forward-inverse consistency seed results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=(11, 12, 13),
    )
    return parser.parse_args()


def mean_std(values):
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=0)),
    }


def aggregate_horizon(runs, horizon):
    keys = (
        "inverse_action_mae",
        "inverse_shuffled_action_mae",
        "inverse_constant_action_mae",
        "cycle_action_mae",
        "action_stride_energy",
    )
    forward_keys = (
        "object_translation_l1",
        "object_rotation_l1",
        "contact_bce",
        "contact_f1",
    )
    result = {
        key: mean_std([
            run["horizons"][horizon][key] for run in runs
        ])
        for key in keys
    }
    for prefix in (
        "forward",
        "forward_history_shuffled",
        "forward_action_shuffled",
    ):
        result[prefix] = {
            key: mean_std([
                run["horizons"][horizon][prefix][key]
                for run in runs
            ])
            for key in forward_keys
        }
    inverse = result["inverse_action_mae"]["mean"]
    shuffled = result["inverse_shuffled_action_mae"]["mean"]
    constant = result["inverse_constant_action_mae"]["mean"]
    result["inverse_relative_improvement_over_shuffle"] = (
        float((shuffled - inverse) / shuffled)
        if shuffled
        else None
    )
    result["inverse_relative_improvement_over_constant"] = (
        float((constant - inverse) / constant)
        if constant
        else None
    )
    forward_translation = result["forward"][
        "object_translation_l1"
    ]["mean"]
    shuffled_translation = result["forward_action_shuffled"][
        "object_translation_l1"
    ]["mean"]
    result["action_shuffle_translation_increase"] = (
        float(
            (shuffled_translation - forward_translation)
            / forward_translation
        )
        if forward_translation
        else None
    )
    result["action_shuffle_contact_f1_drop"] = float(
        result["forward"]["contact_f1"]["mean"]
        - result["forward_action_shuffled"]["contact_f1"]["mean"]
    )
    return result


def main():
    args = parse_args()
    root = Path(args.root).resolve()
    runs = [
        json.loads(
            (root / f"seed_{seed}" / "metrics.json").read_text(
                encoding="utf-8"
            )
        )
        for seed in args.seeds
    ]
    horizons = {
        f"h{index}": aggregate_horizon(runs, f"h{index}")
        for index in range(1, 9)
    }
    gate_horizons = {
        "h4": horizons["h4"],
        "h8": horizons["h8"],
    }
    identifiability_pass = all(
        metrics["inverse_relative_improvement_over_shuffle"] >= 0.05
        and metrics["inverse_relative_improvement_over_constant"] >= 0.0
        and (
            metrics["action_shuffle_translation_increase"] >= 0.05
            or metrics["action_shuffle_contact_f1_drop"] >= 0.02
        )
        for metrics in gate_horizons.values()
    )
    result = {
        "root": str(root),
        "seeds": list(args.seeds),
        "horizons": horizons,
        "identifiability_gate": {
            "pass": bool(identifiability_pass),
            "criteria": {
                "h4_h8_inverse_improvement_over_shuffle": 0.05,
                "h4_h8_inverse_not_worse_than_constant": 0.0,
                "h4_h8_action_shuffle_translation_increase": 0.05,
                "h4_h8_action_shuffle_contact_f1_drop": 0.02,
            },
        },
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "identifiability_gate": result["identifiability_gate"],
        "h4": horizons["h4"],
        "h8": horizons["h8"],
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

