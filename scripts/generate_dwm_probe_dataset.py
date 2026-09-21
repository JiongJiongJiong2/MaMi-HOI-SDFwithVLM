#!/usr/bin/env python3
"""Generate group-level DWM P0 probe and counterfactual data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from manip.world_model.dwm.branches import (
    BRANCH_NAMES,
    FIXED_PROBE_SEQUENCE,
    HORIZON,
    PROBE_HORIZON,
    make_action_branches,
    make_probe_sequences,
    make_target_action,
)
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


PROBE_LENGTHS = (1, 2, 4)
RESULT_VALIDATION_KEYS = (
    "initial_state",
    "probe_action",
    "probe_state",
    "probe_mask",
    "post_probe_state",
    "candidate_action",
    "candidate_final_object_pose",
    "candidate_contact_mode",
    "candidate_contact_impulse",
    "target_action",
    "target_translation",
    "utility",
    "rank",
    "object_id",
    "reset_id",
    "split",
    "probe_mode",
    "probe_length",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=RESET_SEED)
    parser.add_argument("--resets-per-object", type=int, default=120)
    parser.add_argument("--object-limit", type=int, default=0)
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, min(8, os.cpu_count() or 1)),
    )
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def object_index(config):
    digest = hashlib.sha256(config.object_id.encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little")


def probe_seed(base_seed, config, reset_id, mode):
    mode_offset = 0 if mode == "fixed" else 1
    return int(np.random.SeedSequence([
        int(base_seed),
        object_index(config),
        int(reset_id),
        mode_offset,
    ]).generate_state(1, dtype=np.uint32)[0])


def target_seed(base_seed, config, reset_id):
    return int(np.random.SeedSequence([
        int(base_seed),
        object_index(config),
        int(reset_id),
        1000,
    ]).generate_state(1, dtype=np.uint32)[0])


def contact_impulse(states):
    return np.asarray([
        state.contact_forces.sum(axis=0)
        for state in states[1:]
    ], dtype=np.float32)


def rollout_action(env, action_chunk):
    states = [env.snapshot()]
    modes = [CONTACT_MODE_TO_INDEX[env.contact_mode()]]
    for action in action_chunk:
        states.append(env.step(action))
        modes.append(CONTACT_MODE_TO_INDEX[env.contact_mode()])
    return states, np.asarray(modes, dtype=np.int64)


def padded_probe_arrays(probe_actions, probe_states):
    action = np.zeros((PROBE_HORIZON, ACTION_DIM), dtype=np.float32)
    state = np.zeros((PROBE_HORIZON + 1, STATE_DIM), dtype=np.float32)
    mask = np.zeros(PROBE_HORIZON, dtype=np.bool_)
    length = len(probe_actions)
    if length:
        action[:length] = np.asarray(probe_actions, dtype=np.float32)
    state[:length + 1] = np.asarray([
        value.vector for value in probe_states
    ], dtype=np.float32)
    mask[:length] = True
    return action, state, mask


def build_group(
    env,
    checkpoint,
    actions,
    probe_actions,
    probe_states,
    config,
    reset_id,
    split,
    probe_mode,
    target_action,
):
    post_probe = env.restore(checkpoint)
    probe_action, probe_state, probe_mask = padded_probe_arrays(
        probe_actions,
        probe_states,
    )
    candidate_actions = np.stack([
        np.asarray(actions[name], dtype=np.float32)
        for name in BRANCH_NAMES
    ])
    candidate_final_pose = np.zeros((len(BRANCH_NAMES), 9), dtype=np.float32)
    candidate_contact_mode = np.zeros(
        (len(BRANCH_NAMES), HORIZON + 1),
        dtype=np.int64,
    )
    candidate_contact_impulse = np.zeros(
        (len(BRANCH_NAMES), HORIZON, 3),
        dtype=np.float32,
    )
    for index, action in enumerate(candidate_actions):
        env.restore(checkpoint)
        states, modes = rollout_action(env, action)
        candidate_final_pose[index] = states[-1].object_pose
        candidate_contact_mode[index] = modes
        candidate_contact_impulse[index] = contact_impulse(states)

    env.restore(checkpoint)
    target_states, _ = rollout_action(env, target_action)
    target_translation = (
        target_states[-1].object_pose[:3]
        - post_probe.object_pose[:3]
    )
    candidate_translation = (
        candidate_final_pose[:, :3] - post_probe.object_pose[:3]
    )
    utility = -np.sum(
        np.abs(candidate_translation - target_translation[None]),
        axis=-1,
    ).astype(np.float32)
    order = np.argsort(-utility, kind="stable")
    rank = np.empty(len(BRANCH_NAMES), dtype=np.int64)
    rank[order] = np.arange(len(BRANCH_NAMES), dtype=np.int64)

    return {
        "initial_state": np.asarray(
            probe_states[0].vector,
            dtype=np.float32,
        ),
        "probe_action": probe_action,
        "probe_state": probe_state,
        "probe_mask": probe_mask,
        "post_probe_state": np.asarray(
            post_probe.vector,
            dtype=np.float32,
        ),
        "candidate_action": candidate_actions,
        "candidate_final_object_pose": candidate_final_pose,
        "candidate_contact_mode": candidate_contact_mode,
        "candidate_contact_impulse": candidate_contact_impulse,
        "target_action": np.asarray(target_action, dtype=np.float32),
        "target_translation": target_translation.astype(np.float32),
        "utility": utility,
        "rank": rank,
        "object_id": np.asarray(config.object_id),
        "reset_id": np.asarray(reset_id, dtype=np.int64),
        "split": np.asarray(split),
        "probe_mode": np.asarray(probe_mode),
        "probe_length": np.asarray(len(probe_actions), dtype=np.int64),
    }


def collect_config(args, split, config):
    rows = []
    env = DWMSimEnv(config, seed=args.seed)
    for reset_id in range(args.resets_per_object):
        env.reset(reset_id)
        actions = make_action_branches(
            env,
            seed=args.seed + reset_id * 104729,
        )
        initial_checkpoint = env.checkpoint()
        initial_state = env.snapshot()
        target_action = make_target_action(
            env,
            seed=target_seed(args.seed, config, reset_id),
            candidate_actions=actions,
        )
        env.restore(initial_checkpoint)
        rows.append(build_group(
            env,
            initial_checkpoint,
            actions,
            [],
            [initial_state],
            config,
            reset_id,
            split,
            "none",
            target_action,
        ))

        fixed, random = make_probe_sequences(
            actions,
            seed=probe_seed(
                args.seed,
                config,
                reset_id,
                "random",
            ),
        )
        for mode, sequence in (
            ("fixed", fixed),
            ("random", random),
        ):
            env.restore(initial_checkpoint)
            probe_states = [env.snapshot()]
            checkpoint = env.checkpoint()
            for length in PROBE_LENGTHS:
                while len(probe_states) - 1 < length:
                    env.restore(checkpoint)
                    step = len(probe_states) - 1
                    env.step(sequence[step])
                    probe_states.append(env.snapshot())
                    checkpoint = env.checkpoint()
                rows.append(build_group(
                    env,
                    checkpoint,
                    actions,
                    sequence[:length],
                    list(probe_states),
                    config,
                    reset_id,
                    split,
                    mode,
                    target_action,
                ))
        print(
            f"[{split}] {config.object_id} reset={reset_id}",
            flush=True,
        )
    return rows


def collect_split(args, split):
    configs = list(object_configs_for_split(split))
    if args.object_limit > 0:
        configs = configs[:args.object_limit]
    workers = max(1, int(getattr(args, "workers", 1)))
    if workers == 1 or len(configs) == 1:
        rows = []
        for config in configs:
            rows.extend(collect_config(args, split, config))
    else:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(collect_config, args, split, config)
                for config in configs
            ]
            rows = []
            for future in futures:
                rows.extend(future.result())

    result = {}
    for key in RESULT_VALIDATION_KEYS:
        values = [row[key] for row in rows]
        result[key] = np.stack(values)
    result["branch_names"] = np.asarray(BRANCH_NAMES)
    expected_groups = (
        len(configs)
        * args.resets_per_object
        * (1 + 2 * len(PROBE_LENGTHS))
    )
    if result["object_id"].shape != (expected_groups,):
        raise ValueError(
            f"invalid group count: {result['object_id'].shape}"
        )
    if result["candidate_action"].shape[1:] != (13, HORIZON, ACTION_DIM):
        raise ValueError("invalid candidate action shape")
    if result["probe_state"].shape[1:] != (PROBE_HORIZON + 1, STATE_DIM):
        raise ValueError("invalid probe state shape")
    return result


def main():
    args = parse_args()
    if args.resets_per_object < 1:
        raise ValueError("--resets-per-object must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    metadata = {
        "seed": int(args.seed),
        "resets_per_object": int(args.resets_per_object),
        "object_limit": int(args.object_limit),
        "workers": int(args.workers),
        "probe_lengths": list(PROBE_LENGTHS),
        "fixed_probe": list(FIXED_PROBE_SEQUENCE),
        "branch_names": list(BRANCH_NAMES),
        "state_dim": int(STATE_DIM),
        "action_dim": int(ACTION_DIM),
        "splits": {},
    }
    for split in ("train", "val", "test"):
        data = collect_split(args, split)
        output_path = args.output_dir / f"{split}.npz"
        np.savez_compressed(output_path, **data)
        metadata["splits"][split] = {
            "path": str(output_path),
            "sha256": sha256_file(output_path),
            "groups": int(data["object_id"].shape[0]),
            "objects": sorted(set(data["object_id"].tolist())),
            "reset_ids": [
                int(value) for value in sorted(set(data["reset_id"].tolist()))
            ],
        }
    metadata_path = args.output_dir / "metadata.json"
    metadata_path.write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
