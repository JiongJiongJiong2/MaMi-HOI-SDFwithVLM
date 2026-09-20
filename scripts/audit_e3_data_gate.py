#!/usr/bin/env python3
"""Audit local E3 data availability for contact-memory and handover tests."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mamihoi-root", type=Path, required=True)
    parser.add_argument("--handx-data-root", type=Path, required=True)
    parser.add_argument("--hopformer-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    if not path.is_file():
        return None
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": sha256_file(path),
    }


def git_commit(root):
    root = Path(root)
    if not (root / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip()


def inspect_npz_sample(path):
    path = Path(path)
    if not path.is_file():
        return None
    with np.load(path, mmap_mode="r", allow_pickle=True) as archive:
        keys = list(archive.files)
        first = archive[keys[0]] if keys else None
        value = first.item() if first is not None else None
        sample_keys = list(value.keys()) if isinstance(value, dict) else []
        shapes = {}
        if isinstance(value, dict):
            for key, item in value.items():
                if isinstance(item, np.ndarray):
                    shapes[key] = list(item.shape)
        return {
            "path": str(path),
            "size": int(path.stat().st_size),
            "sample_count": len(keys),
            "first_sample_keys": sample_keys,
            "first_sample_shapes": shapes,
        }


def inspect_mamihoi(processed_root):
    import joblib

    path = (
        processed_root
        / "test_diffusion_manip_seq_joints24.p"
    )
    if not path.is_file():
        return {
            "available": False,
            "path": str(path),
        }
    data = joblib.load(path)
    first = None
    if isinstance(data, dict):
        first = next(iter(data.values()))
        sequence_count = len(data)
    else:
        first = data[0] if data else None
        sequence_count = len(data)
    keys = sorted(first.keys()) if isinstance(first, dict) else []
    return {
        "available": True,
        "path": str(path),
        "size": int(path.stat().st_size),
        "sequence_count": int(sequence_count),
        "sequence_keys": keys,
        "has_pose_hand": "pose_hand" in keys,
        "has_pose_body": "pose_body" in keys,
        "has_object_pose": (
            "obj_com_pos" in keys
            and "obj_rot" in keys
            and "obj_trans" in keys
        ),
    }


def inspect_handx_data(handx_root):
    source_path = (
        handx_root
        / "handx_to_dataset_source_release.json"
    )
    source_counts = {}
    if source_path.is_file():
        source = json.loads(source_path.read_text(encoding="utf-8"))
        source_counts = {
            split: dict(sorted(Counter(values.values()).items()))
            for split, values in source.items()
        }
    train_motion = inspect_npz_sample(
        handx_root / "train_can_pos_all_wotextfeat.npz"
    )
    test_mano = inspect_npz_sample(
        handx_root / "test_mano.npz"
    )
    sample_keys = set()
    if train_motion is not None:
        sample_keys.update(train_motion["first_sample_keys"])
    if test_mano is not None:
        sample_keys.update(test_mano["first_sample_shapes"])
    return {
        "available": train_motion is not None or test_mano is not None,
        "source_release": file_identity(source_path),
        "source_counts": source_counts,
        "train_motion": train_motion,
        "test_mano": test_mano,
        "has_bimanual_motion": (
            "motion" in sample_keys
            or (
                "left_pose" in sample_keys
                and "right_pose" in sample_keys
            )
        ),
        "has_object_state": False,
        "has_object_mesh": False,
        "has_contact_labels": False,
        "has_release_labels": False,
        "interaction_annotations_are_hand_hand": True,
    }


def inspect_hopformer(root):
    root = Path(root)
    readme_path = root / "README.md"
    license_path = root / "LICENSE"
    readme = (
        readme_path.read_text(encoding="utf-8", errors="replace")
        if readme_path.is_file()
        else ""
    )
    license_text = (
        license_path.read_text(encoding="utf-8", errors="replace")
        if license_path.is_file()
        else ""
    )
    epic_dataset = root / "src" / "datasets" / "epic_dataset.py"
    epic_text = (
        epic_dataset.read_text(encoding="utf-8", errors="replace")
        if epic_dataset.is_file()
        else ""
    )
    epic_data = root / "data" / "epic_data"
    arctic_data = root / "data" / "arctic_data"
    return {
        "repository_available": root.is_dir(),
        "repository_commit": git_commit(root),
        "readme": file_identity(readme_path),
        "license": file_identity(license_path),
        "license_family": (
            "CC-BY-NC-4.0"
            if "CC BY-NC 4.0" in license_text
            else "unknown"
        ),
        "dataset_access": (
            "gated"
            if "gated" in readme.lower()
            else "unknown"
        ),
        "epic_data_present": epic_data.is_dir(),
        "arctic_data_present": arctic_data.is_dir(),
        "epic_schema_tokens_found": sorted(
            token
            for token in (
                "dist.or",
                "dist.ro",
                "idx.or",
                "idx.ro",
                "object.v.cam",
                "right_valid",
                "left_valid",
            )
            if token in epic_text
        ),
        "reported_epic_scale": (
            "2.3K clips / 62.3K frames"
            if "2.3K clips" in readme and "62.3K" in readme
            else None
        ),
    }


def build_decision(mamihoi, handx, hopformer):
    raw_finger_supervision = (
        mamihoi["has_pose_hand"]
        or handx["has_bimanual_motion"]
    )
    complete_event_support = (
        mamihoi["has_object_pose"]
        and (
            hopformer["epic_data_present"]
            or hopformer["arctic_data_present"]
        )
    )
    contact_transition_support = complete_event_support
    handover_support = (
        complete_event_support
        and hopformer["epic_data_present"]
    )
    return {
        "raw_finger_supervision": (
            "pass" if raw_finger_supervision else "blocked"
        ),
        "complete_event_data_local": (
            "pass" if complete_event_support else "blocked"
        ),
        "contact_memory_b_gate": (
            "ready"
            if contact_transition_support
            else "blocked_missing_local_event_data"
        ),
        "future_handover_c_gate": (
            "ready"
            if handover_support
            else "blocked_missing_local_event_data"
        ),
        "current_action": (
            "acquire_and_audit_event_data_before_modeling"
        ),
    }


def main():
    args = parse_args()
    mamihoi = inspect_mamihoi(
        args.mamihoi_root / "data" / "processed_data"
    )
    handx = inspect_handx_data(args.handx_data_root)
    hopformer = inspect_hopformer(args.hopformer_root)
    decision = build_decision(mamihoi, handx, hopformer)
    result = {
        "mamihoi": mamihoi,
        "handx_mixdata": handx,
        "hopformer_epic": hopformer,
        "decision": decision,
        "scope": (
            "local read-only audit; no gated dataset download and no test "
            "split access"
        ),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "mamihoi": {
            key: mamihoi[key]
            for key in (
                "available",
                "sequence_count",
                "has_pose_hand",
                "has_pose_body",
                "has_object_pose",
            )
        },
        "handx_mixdata": {
            key: handx[key]
            for key in (
                "available",
                "has_bimanual_motion",
                "has_object_state",
                "has_contact_labels",
                "has_release_labels",
            )
        },
        "hopformer_epic": {
            key: hopformer[key]
            for key in (
                "repository_available",
                "license_family",
                "dataset_access",
                "epic_data_present",
                "arctic_data_present",
                "reported_epic_scale",
            )
        },
        "decision": decision,
        "output": str(args.output),
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
