#!/usr/bin/env python3
"""Evaluate ContactOpt input/refined hand-object contact and distances."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
from scipy.spatial.distance import cdist

from contactopt_contact_metrics import sanitized_contact_mean


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("optimized_pkl", type=Path)
    parser.add_argument(
        "--contactopt-root",
        type=Path,
        default=Path("/root/autodl-tmp/external/ContactOpt"),
    )
    parser.add_argument("--sample", type=int, default=0)
    return parser.parse_args()


def summarize_hand_object(hand_object):
    hand_object.calc_dist_contact(hand=True, obj=True)
    hand_contact_mean, hand_contact_finite = sanitized_contact_mean(
        hand_object.hand_contact
    )
    obj_contact_mean, obj_contact_finite = sanitized_contact_mean(
        hand_object.obj_contact
    )
    distances = cdist(hand_object.hand_verts, hand_object.obj_verts)

    return {
        "hand_contact_mean": hand_contact_mean,
        "object_contact_mean": obj_contact_mean,
        "hand_contact_finite_fraction": hand_contact_finite,
        "object_contact_finite_fraction": obj_contact_finite,
        "mean_nearest_distance_m": float(distances.min(axis=1).mean()),
        "minimum_distance_m": float(distances.min()),
        "hand_vertices": int(hand_object.hand_verts.shape[0]),
        "object_vertices": int(hand_object.obj_verts.shape[0]),
    }


def main():
    args = parse_args()
    sys.path.insert(0, str(args.contactopt_root))

    with args.optimized_pkl.open("rb") as handle:
        runs = pickle.load(handle)

    sample = runs[args.sample]
    result = {
        "optimized_pkl": str(args.optimized_pkl),
        "sample": args.sample,
        "input": summarize_hand_object(sample["in_ho"]),
        "refined": summarize_hand_object(sample["out_ho"]),
    }

    input_distance = result["input"]["mean_nearest_distance_m"]
    refined_distance = result["refined"]["mean_nearest_distance_m"]
    result["distance_relative_change"] = (
        refined_distance / input_distance - 1.0
        if input_distance > 0
        else None
    )

    input_contact = result["input"]["hand_contact_mean"]
    refined_contact = result["refined"]["hand_contact_mean"]
    result["hand_contact_relative_change"] = (
        refined_contact / input_contact - 1.0 if input_contact > 0 else None
    )

    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
