#!/usr/bin/env python3
"""Evaluate the frozen direction-projection arms against fixed controls."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


CONTROL_ARMS = (
    "ordinary_smoothing",
    "contact_projection",
    "strong_contact",
)
ALL_ARMS = (
    "ordinary_smoothing",
    "contact_projection",
    "unconstrained_continuation",
    "strong_contact",
    "direction_constrained",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-root", type=Path, required=True)
    parser.add_argument("--e5-result", type=Path, required=True)
    parser.add_argument("--e5t-result", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--bootstrap-samples", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=20260922)
    return parser.parse_args()


def window_key(row):
    return (
        row["chunk_id"],
        tuple(int(frame) for frame in row["frames"]),
    )


def finite_mean(values):
    values = np.asarray(values, dtype=np.float64)
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def load_result(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def result_rows(result):
    return {window_key(row): row for row in result["windows"]}


def validate_window_sets(reference, candidates):
    reference_keys = set(reference)
    for name, rows in candidates.items():
        if set(rows) != reference_keys:
            raise ValueError(f"{name} window set differs from E5")


def row_metrics(row):
    contact = row["optimized_contact_gate"]
    distance_change = contact.get("mean_distance_relative_change")
    return {
        "sequence": row["sequence"],
        "split": row["split"],
        "object_name": row["object_name"],
        "combined_pass": bool(row["optimized_combined_pass"]),
        "contact_pass": bool(contact["passed"]),
        "temporal_pass": bool(
            row["optimized_temporal_gate"]["passed"]
        ),
        "acceleration_ratio": row.get("optimized_acceleration_ratio"),
        "jerk_ratio": row.get("optimized_jerk_ratio"),
        "distance_change": distance_change,
        "normalized_distance": (
            1.0 + distance_change
            if distance_change is not None
            else None
        ),
    }


def split_rows(rows, split):
    if split == "all":
        return list(rows)
    return [row for row in rows if row["split"] == split]


def arm_summary(rows, split):
    selected = split_rows(rows, split)
    return {
        "window_count": len(selected),
        "sequence_count": len({
            row["sequence"] for row in selected
        }),
        "combined_pass_count": int(sum(
            row["combined_pass"] for row in selected
        )),
        "contact_pass_count": int(sum(
            row["contact_pass"] for row in selected
        )),
        "temporal_pass_count": int(sum(
            row["temporal_pass"] for row in selected
        )),
        "mean_acceleration_ratio": finite_mean([
            row["acceleration_ratio"] for row in selected
        ]),
        "mean_jerk_ratio": finite_mean([
            row["jerk_ratio"] for row in selected
        ]),
        "mean_distance_change": finite_mean([
            row["distance_change"] for row in selected
        ]),
        "mean_normalized_distance": finite_mean([
            row["normalized_distance"] for row in selected
        ]),
    }


def cluster_bootstrap_count_difference(
    candidate,
    baseline,
    groups,
    samples,
    seed,
):
    candidate = np.asarray(candidate, dtype=np.float64)
    baseline = np.asarray(baseline, dtype=np.float64)
    groups = np.asarray(groups)
    if not (
        candidate.shape == baseline.shape
        == groups.shape
    ):
        raise ValueError("candidate, baseline, and groups must align")
    unique_groups, inverse = np.unique(groups, return_inverse=True)
    indices = [
        np.flatnonzero(inverse == index)
        for index in range(len(unique_groups))
    ]
    observed = int(candidate.sum() - baseline.sum())
    if samples <= 0:
        return {"mean": observed, "ci95": None}
    rng = np.random.default_rng(seed)
    draws = np.empty(samples, dtype=np.int64)
    for draw in range(samples):
        sampled = rng.integers(
            0,
            len(unique_groups),
            size=len(unique_groups),
        )
        chosen = np.concatenate([
            indices[index] for index in sampled
        ])
        draws[draw] = int(
            candidate[chosen].sum()
            - baseline[chosen].sum()
        )
    return {
        "mean": observed,
        "ci95": [
            int(np.quantile(draws, 0.025)),
            int(np.quantile(draws, 0.975)),
        ],
    }


def paired_count_difference(
    rows,
    candidate_name,
    baseline_name,
    split,
    samples,
    seed,
):
    selected = split_rows(rows, split)
    return cluster_bootstrap_count_difference(
        [row[candidate_name]["combined_pass"] for row in selected],
        [row[baseline_name]["combined_pass"] for row in selected],
        [row["sequence"] for row in selected],
        samples,
        seed,
    )


def summarize_result(
    arm_rows,
    e5_rows,
    e5t_rows,
    bootstrap_samples,
    seed,
):
    rows = []
    for key in sorted(e5_rows):
        rows.append({
            "key": key,
            "sequence": e5_rows[key]["sequence"],
            "split": e5_rows[key]["split"],
            "object_name": e5_rows[key]["object_name"],
            "e5": row_metrics(e5_rows[key]),
            "e5t": row_metrics(e5t_rows[key]),
            **{
                arm: row_metrics(arm_rows[arm][key])
                for arm in ALL_ARMS
            },
        })

    arm_names = ("e5", "e5t", *ALL_ARMS)
    summaries = {
        split: {
            arm: arm_summary(
                [
                    {
                        "sequence": row["sequence"],
                        "split": row["split"],
                        **row[arm],
                    }
                    for row in rows
                ],
                split,
            )
            for arm in arm_names
        }
        for split in ("train", "dev", "all")
    }
    differences = {}
    offset = 0
    for split in ("train", "dev", "all"):
        differences[split] = {}
        for control in ("e5t", *CONTROL_ARMS):
            differences[split][control] = paired_count_difference(
                rows,
                "direction_constrained",
                control,
                split,
                bootstrap_samples,
                seed + offset,
            )
            offset += 1
    return rows, summaries, differences


def control_ci_gate(differences):
    controls = {}
    for control in CONTROL_ARMS:
        control_differences = {
            split: differences[split][control]
            for split in ("train", "dev", "all")
        }
        split_checks = {
            split: (
                item["ci95"] is not None
                and item["ci95"][0] > 0
            )
            for split, item in control_differences.items()
        }
        controls[control] = {
            "differences": control_differences,
            "lower_bound_positive": split_checks,
            "lower_bound_positive_on_train_and_dev": all(
                split_checks[split]
                for split in ("train", "dev")
            ),
        }
    return {
        "controls": controls,
        "any_control_ci_lower_positive": any(
            any(item["lower_bound_positive"].values())
            for item in controls.values()
        ),
        "any_control_ci_lower_positive_train_and_dev": any(
            item["lower_bound_positive_on_train_and_dev"]
            for item in controls.values()
        ),
    }


def not_above(candidate, baseline):
    return (
        candidate is not None
        and baseline is not None
        and candidate <= baseline
    )


def acceptance_gate(summaries, differences):
    direction = summaries["all"]["direction_constrained"]
    e5 = summaries["all"]["e5"]
    e5t = summaries["all"]["e5t"]
    control_gate = control_ci_gate(differences)
    contact_regression = {
        split: (
            summaries[split]["e5"]["contact_pass_count"]
            - summaries[split]["direction_constrained"][
                "contact_pass_count"
            ]
        )
        for split in ("train", "dev", "all")
    }
    distance_ratio = (
        direction["mean_normalized_distance"]
        / e5["mean_normalized_distance"]
        if (
            direction["mean_normalized_distance"] is not None
            and e5["mean_normalized_distance"] not in (None, 0.0)
        )
        else None
    )
    checks = {
        "combined_gain_over_e5t_train": (
            summaries["train"]["direction_constrained"][
                "combined_pass_count"
            ]
            > summaries["train"]["e5t"]["combined_pass_count"]
        ),
        "combined_gain_over_e5t_dev": (
            summaries["dev"]["direction_constrained"][
                "combined_pass_count"
            ]
            > summaries["dev"]["e5t"]["combined_pass_count"]
        ),
        "contact_not_below_e5t_train_and_dev": all(
            summaries[split]["direction_constrained"][
                "contact_pass_count"
            ]
            >= summaries[split]["e5t"]["contact_pass_count"]
            for split in ("train", "dev")
        ),
        "contact_regression_vs_e5_within_12": all(
            value <= 12 for value in contact_regression.values()
        ),
        "temporal_not_below_e5t_train_and_dev": all(
            summaries[split]["direction_constrained"][
                "temporal_pass_count"
            ]
            >= summaries[split]["e5t"]["temporal_pass_count"]
            for split in ("train", "dev")
        ),
        "control_ci_lower_positive": (
            control_gate["any_control_ci_lower_positive"]
        ),
        "mean_acceleration_not_above_e5t": not_above(
            direction["mean_acceleration_ratio"],
            e5t["mean_acceleration_ratio"],
        ),
        "mean_jerk_not_above_e5t": not_above(
            direction["mean_jerk_ratio"],
            e5t["mean_jerk_ratio"],
        ),
        "mean_distance_within_10pct_of_e5": (
            distance_ratio is not None
            and distance_ratio <= 1.10
        ),
        "test_untouched": True,
    }
    checks["pass"] = all(
        value for key, value in checks.items() if key != "pass"
    )

    strict_control_checks = {}
    for control in CONTROL_ARMS:
        item = differences["all"][control]
        strict_control_checks[control] = (
            item["mean"] > 0
            and item["ci95"] is not None
            and item["ci95"][0] > 0
        )
    mechanism_gate = {
        "strictly_above_all_fixed_controls": all(
            strict_control_checks.values()
        ),
        "control_checks": strict_control_checks,
    }
    if not checks["pass"]:
        contact_regression_removed = (
            checks["contact_not_below_e5t_train_and_dev"]
            and checks["contact_regression_vs_e5_within_12"]
        )
        decision = (
            "NO-GO promotion: contact regression removed, mechanism not "
            "established; do not retune on dev"
            if contact_regression_removed
            else (
                "NO-GO direction explanation; contact regression remains; "
                "proceed to B"
            )
        )
    elif mechanism_gate["strictly_above_all_fixed_controls"]:
        decision = "GO direction mechanism"
    else:
        decision = "ENGINEERING_ONLY direction correction"
    return {
        "checks": checks,
        "contact_regression_vs_e5": contact_regression,
        "distance_ratio_vs_e5": distance_ratio,
        "control_gate": control_gate,
        "mechanism_gate": mechanism_gate,
        "decision": decision,
    }


def main():
    args = parse_args()
    e5_rows = result_rows(load_result(args.e5_result))
    e5t_rows = result_rows(load_result(args.e5t_result))
    if set(e5_rows) != set(e5t_rows):
        raise ValueError("E5 and E5-T window sets differ")
    arm_rows = {}
    for arm in ALL_ARMS:
        path = args.result_root / arm / "result.json"
        if not path.is_file():
            raise FileNotFoundError(path)
        arm_rows[arm] = result_rows(load_result(path))
    validate_window_sets(e5_rows, arm_rows)

    rows, summaries, differences = summarize_result(
        arm_rows,
        e5_rows,
        e5t_rows,
        args.bootstrap_samples,
        args.seed,
    )
    gate = acceptance_gate(summaries, differences)
    result = {
        "method": (
            "paired train/dev comparison of direction-constrained temporal "
            "correction against ordinary smoothing, contact projection, "
            "strong contact, and frozen E5/E5-T"
        ),
        "bootstrap_samples": args.bootstrap_samples,
        "seed": args.seed,
        "result_root": str(args.result_root),
        "summaries": summaries,
        "paired_combined_pass_differences": differences,
        "acceptance": gate,
        "windows": [
            {
                **row,
                "e5": {
                    "combined_pass": row["e5"]["combined_pass"],
                    "contact_pass": row["e5"]["contact_pass"],
                    "temporal_pass": row["e5"]["temporal_pass"],
                },
                "e5t": {
                    "combined_pass": row["e5t"]["combined_pass"],
                    "contact_pass": row["e5t"]["contact_pass"],
                    "temporal_pass": row["e5t"]["temporal_pass"],
                },
                "direction_constrained": {
                    "combined_pass": row["direction_constrained"][
                        "combined_pass"
                    ],
                    "contact_pass": row["direction_constrained"][
                        "contact_pass"
                    ],
                    "temporal_pass": row["direction_constrained"][
                        "temporal_pass"
                    ],
                },
            }
            for row in rows
        ],
        "test_split": "untouched",
    }
    result["decision"] = gate["decision"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "decision": result["decision"],
        "summaries": summaries,
        "acceptance_checks": gate["checks"],
        "mechanism_gate": gate["mechanism_gate"],
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
