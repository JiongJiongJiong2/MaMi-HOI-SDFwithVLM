#!/usr/bin/env python3
"""Run frozen HandX generation conditioned on MaMi wrist trajectories."""

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf


WRIST_JOINTS = (0, 21)
MAMI_WRIST_JOINTS = (20, 21)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--handx-root", type=Path, required=True)
    parser.add_argument("--checkpoint-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--candidate-npz", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--start-frame", type=int, default=30)
    parser.add_argument("--num-frames", type=int, default=60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--use-fp16", action="store_true")
    parser.add_argument("--hard-mask", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    diffusion_root = args.handx_root / "diffusion"
    sys.path.insert(0, str(diffusion_root))

    from scripts.evaluation.evaluate_val_samples import (
        generate_samples_batch,
        load_checkpoint_config,
        load_model_and_diffusion,
    )
    from src.diffusion.data_loader.handx import HandXDataset
    from src.diffusion.model.cls_free_sampler import (
        ClassifierFreeSampleWrapper,
    )
    from src.diffusion.utils.mics import fixseed, get_device

    config = load_checkpoint_config(str(args.checkpoint_dir))
    config.data.data_dir = str(args.data_dir)
    config.data.data_file_name = "can_pos_all_wotextfeat.npz"
    config.data.repr = "joint_pos_w_scalar_rot"
    config.data.normalize = True
    config.data.contact_label = False
    config.data.ratio = 1.0

    dataset = HandXDataset(
        split="val",
        data_dir=str(args.data_dir),
        data_file_name=config.data.data_file_name,
        repr=config.data.repr,
        normalize=True,
        contact_label=False,
        ratio=1.0,
    )
    _, _, text = dataset[0]

    candidate = np.load(args.candidate_npz, allow_pickle=True)
    global_jpos = np.asarray(candidate["global_jpos"], dtype=np.float64)
    start = args.start_frame
    end = start + args.num_frames
    if end > len(global_jpos):
        raise ValueError(
            f"Requested frames {start}:{end}, but trajectory has "
            f"{len(global_jpos)} frames"
        )

    wrists = global_jpos[start:end, MAMI_WRIST_JOINTS, :]
    translation = wrists.reshape(-1, 3).mean(axis=0)
    wrists = wrists - translation

    raw_motion = np.zeros(
        (args.num_frames, 42, 4),
        dtype=np.float64,
    )
    raw_motion[:, 0, :3] = wrists[:, 0]
    raw_motion[:, 21, :3] = wrists[:, 1]
    normalized = (raw_motion - dataset.mean) / dataset.std
    normalized = normalized.reshape(args.num_frames, -1).astype(
        np.float32
    )

    val_sample = {
        "index": -1,
        "text": text,
        "motion": normalized,
        "length": args.num_frames,
    }
    mask_region = {
        "temporal": list(range(args.num_frames)),
        "spatial": list(WRIST_JOINTS),
    }
    if not args.hard_mask:
        mask_region.update(
            {
                "use_soft_mask": True,
                "temporal_transition_width": 5,
                "core_mask_value": 0.85,
                "edge_mask_value": 0.1,
            }
        )
    mask_regions = [mask_region]

    model, diffusion = load_model_and_diffusion(
        str(args.checkpoint_dir / "model.pt"),
        config,
    )
    device = get_device()
    model.to(device)
    model.eval()
    wrapped = ClassifierFreeSampleWrapper(model, scale=2.5)
    fixseed(args.seed)

    _, generated_real = generate_samples_batch(
        model=wrapped,
        diffusion=diffusion,
        dataset=dataset,
        val_samples_batch=[val_sample],
        num_samples_per_text=1,
        guidance_scale=2.5,
        njoints=42,
        nfeats=4,
        fixed_frames=None,
        mask_regions=mask_regions,
        use_fp16=args.use_fp16,
    )[0]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    sample_path = args.output_dir / "val_sample_000_idxmami.pkl"
    with sample_path.open("wb") as handle:
        pickle.dump(
            {
                "text_prompt": text,
                "val_index": -1,
                "motion_length": args.num_frames,
                "gt_motion_real": raw_motion.reshape(
                    args.num_frames, -1
                ),
                "generated_real": np.asarray(
                    generated_real, dtype=np.float64
                ),
                "njoints": 42,
                "nfeats": 4,
                "mask_regions": mask_regions,
                "candidate_npz": str(args.candidate_npz),
                "start_frame": start,
                "end_frame": end,
            },
            handle,
        )
    print(
        json.dumps(
            {
                "sample": str(sample_path),
                "shape": list(np.asarray(generated_real).shape),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
