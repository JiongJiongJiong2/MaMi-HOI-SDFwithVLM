#!/usr/bin/env python3
"""Print geometry diagnostics for HandX-to-MANO projection."""

import argparse
import pickle
import sys
from pathlib import Path

import numpy as np

from project_handx_to_mano import (
    ARTICULATED_CHAIN,
    extract_generated_motion,
    initialize_world_transform,
    make_mano_layer,
    mano_forward,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=Path, required=True)
    parser.add_argument("--contactopt-root", type=Path, required=True)
    parser.add_argument("--mano-root", type=Path, required=True)
    return parser.parse_args()


def lengths(joints, chains):
    values = []
    for chain in chains:
        values.append(
            np.linalg.norm(
                joints[:, chain[1:]] - joints[:, chain[:-1]],
                axis=2,
            )
        )
    return np.concatenate(values, axis=1)


def main():
    args = parse_args()
    sys.path.insert(0, str(args.contactopt_root))
    import torch

    with args.sample.open("rb") as handle:
        sample = pickle.load(handle)
    motion = extract_generated_motion(sample["generated_real"])
    for side_index, side in enumerate(("left", "right")):
        target = motion[:, side_index * 21 : (side_index + 1) * 21]
        layer = make_mano_layer(
            args.contactopt_root, args.mano_root, side
        ).cuda()
        with torch.no_grad():
            _, neutral = mano_forward(
                layer,
                torch.zeros(1, 48, device="cuda"),
                torch.zeros(1, 10, device="cuda"),
                torch.zeros(1, 3, device="cuda"),
            )
        neutral = (neutral[0] / 1000.0).cpu().numpy()
        rotations, translations = initialize_world_transform(
            neutral, target
        )
        aligned = (
            np.einsum("bij,kj->bki", rotations, neutral)
            + translations[:, None]
        )
        print(side)
        print("target wrist", target[0, 0])
        print("neutral wrist", neutral[0])
        print("aligned wrist", aligned[0, 0])
        print("target lengths", lengths(target, ARTICULATED_CHAIN).mean(0))
        print(
            "neutral lengths",
            lengths(neutral[None], ARTICULATED_CHAIN)[0],
        )
        print(
            "neutral-aligned error mm",
            np.linalg.norm(aligned[0] - target[0], axis=1) * 1000.0,
        )


if __name__ == "__main__":
    main()
