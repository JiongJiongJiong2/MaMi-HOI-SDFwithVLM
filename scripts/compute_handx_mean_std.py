#!/usr/bin/env python3
"""Compute HandX joint_pos_w_scalar_rot normalization from an NPZ or zip."""

import argparse
import sys
import zipfile
from pathlib import Path

import numpy as np
from tqdm import tqdm


def parse_args():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--npz", type=Path)
    source.add_argument("--zip", type=Path)
    parser.add_argument("--member")
    parser.add_argument("--handx-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def compute_scalar_rotation(motion, side, motion_coder_class, chains):
    coder = motion_coder_class(motion, isright=(side == "right"))
    coder.get_local_coordinate()
    local_motion = coder.local_motion
    scalar = np.zeros((motion.shape[0], motion.shape[1]), dtype=np.float64)
    for chain in chains:
        scalar[:, chain[0]] = 0
        scalar[:, chain[-1]] = 0
        for index in range(1, len(chain) - 1):
            joint = chain[index]
            previous = chain[index - 1]
            following = chain[index + 1]
            vector_a = (
                local_motion[:, joint, :] - local_motion[:, previous, :]
            )[:, [0, 2]]
            vector_b = (
                local_motion[:, following, :] - local_motion[:, joint, :]
            )[:, [0, 2]]
            angle_a = np.arctan2(vector_a[:, 1], vector_a[:, 0])
            angle_b = np.arctan2(vector_b[:, 1], vector_b[:, 0])
            angle = angle_b - angle_a
            if side == "right":
                angle = -angle
            scalar[:, joint] = angle
    return scalar


def main():
    args = parse_args()
    diffusion_root = args.handx_root / "diffusion"
    sys.path.insert(0, str(diffusion_root))
    from src.constant import SKELETON_CHAIN
    from src.feature.single_motioncode import MotionCoder

    if args.npz is not None:
        handle = args.npz.open("rb")
        close_handle = True
    else:
        archive = zipfile.ZipFile(args.zip)
        member = args.member
        if member is None:
            raise ValueError("--member is required with --zip")
        handle = archive.open(member)
        close_handle = True

    try:
        data = np.load(handle, allow_pickle=True)
        keys = list(data.files)
        feature_shape = (42, 4)
        total = 0
        sum_values = np.zeros(feature_shape, dtype=np.float64)
        sum_squares = np.zeros(feature_shape, dtype=np.float64)

        for key in tqdm(keys, desc="Computing HandX mean/std"):
            sample = data[key].item()
            motion = np.asarray(sample["motion"], dtype=np.float64)
            left = compute_scalar_rotation(
                motion[:, 0],
                "left",
                MotionCoder,
                SKELETON_CHAIN,
            )
            right = compute_scalar_rotation(
                motion[:, 1],
                "right",
                MotionCoder,
                SKELETON_CHAIN,
            )
            features = np.concatenate(
                [
                    motion,
                    np.stack([left, right], axis=1)[:, :, :, None],
                ],
                axis=-1,
            ).reshape(motion.shape[0], -1, 4)
            translation = np.mean(
                (features[:, 0, :3] + features[:, 21, :3]) / 2.0,
                axis=0,
            )
            features[:, :, :3] -= translation
            features = np.nan_to_num(
                features,
                nan=0.0,
                posinf=0.0,
                neginf=0.0,
            )
            total += features.shape[0]
            sum_values += features.sum(axis=0)
            sum_squares += np.square(features).sum(axis=0)

        mean = sum_values / total
        variance = sum_squares / total - np.square(mean)
        variance[variance < 0] = 0
        std = np.sqrt(variance)
        std[std < 1e-4] = 1.0

        args.output_dir.mkdir(parents=True, exist_ok=True)
        np.save(args.output_dir / "mean.npy", mean)
        np.save(args.output_dir / "std.npy", std)
        print(
            {
                "clips": len(keys),
                "frames": total,
                "mean_shape": mean.shape,
                "std_shape": std.shape,
                "output_dir": str(args.output_dir),
            }
        )
    finally:
        if close_handle and handle is not None:
            handle.close()
        if args.zip is not None:
            archive.close()


if __name__ == "__main__":
    main()
