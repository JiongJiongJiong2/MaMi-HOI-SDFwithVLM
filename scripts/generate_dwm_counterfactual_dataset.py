#!/usr/bin/env python3
"""Generate DWM same-state counterfactual MuJoCo trajectories."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from manip.world_model.dwm.branches import BRANCH_NAMES, make_action_branches
from manip.world_model.dwm.config import (
    ACTION_DIM,
    RESET_SEED,
    object_configs_for_split,
)
from manip.world_model.dwm.env import DWMSimEnv
from manip.world_model.dwm.schema import (
    CONTACT_MODE_TO_INDEX,
    STATE_DIM,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=RESET_SEED)
    parser.add_argument("--resets-per-object", type=int, default=120)
    parser.add_argument("--object-limit", type=int, default=0)
    parser.add_argument("--horizon", type=int, default=8)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def collect_split(args, split):
    configs = list(object_configs_for_split(split))
    if args.object_limit > 0:
        configs = configs[:args.object_limit]
    state_rows = []
    action_rows = []
    mode_rows = []
    reset_rows = []
    object_rows = []
    branch_rows = []
    for config_index, config in enumerate(configs):
        env = DWMSimEnv(config, seed=args.seed)
        for reset_id in range(args.resets_per_object):
            env.reset(reset_id)
            actions = make_action_branches(env, seed=args.seed + reset_id)
            for branch_name in BRANCH_NAMES:
                action_chunk = actions[branch_name][:args.horizon]
                env.reset(reset_id)
                states = [env.snapshot().vector]
                modes = [CONTACT_MODE_TO_INDEX[env.contact_mode()]]
                for action in action_chunk:
                    state = env.step(action)
                    states.append(state.vector)
                    modes.append(CONTACT_MODE_TO_INDEX[env.contact_mode()])
                state_rows.append(np.stack(states))
                action_rows.append(action_chunk.astype(np.float32))
                mode_rows.append(np.asarray(modes, dtype=np.int64))
                reset_rows.append(reset_id)
                object_rows.append(config.object_id)
                branch_rows.append(branch_name)
                print(
                    f"[{split}] {config.object_id} reset={reset_id} "
                    f"branch={branch_name}",
                    flush=True,
                )
    state_array = np.stack(state_rows)
    action_array = np.stack(action_rows)
    mode_array = np.stack(mode_rows)
    if state_array.shape[1:] != (args.horizon + 1, STATE_DIM):
        raise ValueError(f"invalid state shape: {state_array.shape}")
    if action_array.shape[1:] != (args.horizon, ACTION_DIM):
        raise ValueError(f"invalid action shape: {action_array.shape}")
    return {
        "states": state_array,
        "actions": action_array,
        "object_pose": state_array[:, :, 0:9],
        "object_twist": state_array[:, :, 9:15],
        "contact_forces": state_array[:, :, 120:168].reshape(
            state_array.shape[0],
            args.horizon + 1,
            16,
            3,
        ),
        "contact_mode": mode_array,
        "reset_id": np.asarray(reset_rows, dtype=np.int64),
        "object_id": np.asarray(object_rows),
        "branch_name": np.asarray(branch_rows),
        "split": np.full(len(reset_rows), split),
    }


def main():
    args = parse_args()
    if args.resets_per_object < 1:
        raise ValueError("--resets-per-object must be positive")
    if args.horizon < 1:
        raise ValueError("--horizon must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": int(args.seed),
        "resets_per_object": int(args.resets_per_object),
        "object_limit": int(args.object_limit),
        "horizon": int(args.horizon),
        "state_dim": int(STATE_DIM),
        "action_dim": int(ACTION_DIM),
        "branch_names": list(BRANCH_NAMES),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        data = collect_split(args, split)
        output_path = args.output_dir / f"{split}.npz"
        np.savez_compressed(output_path, **data)
        metadata["splits"][split] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "trajectories": int(data["states"].shape[0]),
            "objects": sorted(set(data["object_id"].tolist())),
            "resets": sorted(set(data["reset_id"].tolist())),
        }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
