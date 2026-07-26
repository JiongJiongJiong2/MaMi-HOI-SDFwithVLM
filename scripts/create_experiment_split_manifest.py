"""Create a deterministic, sequence-disjoint validation/test manifest.

The original code has one held-out subject split and uses it for both
validation and test.  This script partitions its unique sequence names without
changing any motion/SDF data, preventing checkpoint selection on test names.
"""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

import joblib


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output_path", required=True)
    parser.add_argument("--validation_sequence_count", type=int, default=100)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--window", type=int, default=120)
    return parser.parse_args()


def object_name(sequence_name):
    parts = sequence_name.split("_")
    return parts[1] if len(parts) > 1 else "unknown"


def counts(names):
    return dict(sorted(Counter(object_name(name) for name in names).items()))


def main():
    args = parse_args()
    data_root = Path(args.data_root_folder)
    processed_path = data_root / (
        f"cano_test_diffusion_manip_window_{args.window}_joints24.p"
    )
    if not processed_path.exists():
        raise FileNotFoundError(
            f"Missing {processed_path}. Run normal MaMi-HOI preprocessing first."
        )
    data = joblib.load(processed_path)
    eligible_names = sorted(
        {
            item["seq_name"]
            for item in data.values()
            if len(item["motion"]) >= args.window
            and int(item.get("start_t_idx", 0)) == 0
        }
    )
    if args.validation_sequence_count <= 0:
        raise ValueError("validation_sequence_count must be positive")
    if len(eligible_names) <= args.validation_sequence_count:
        raise ValueError(
            f"Need more than {args.validation_sequence_count} eligible sequences, "
            f"found {len(eligible_names)}"
        )

    def rank(name):
        payload = f"{args.seed}:{name}".encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    ranked = sorted(eligible_names, key=rank)
    validation = sorted(ranked[: args.validation_sequence_count])
    test = sorted(ranked[args.validation_sequence_count :])
    if set(validation).intersection(test):
        raise AssertionError("validation/test overlap")
    manifest = {
        "format_version": 1,
        "source_file": str(processed_path),
        "seed": args.seed,
        "window": args.window,
        "selection_rule": "ascending sha256('<seed>:<sequence_name>')",
        "validation_sequences": validation,
        "test_sequences": test,
        "validation_object_counts": counts(validation),
        "test_object_counts": counts(test),
    }
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "output_path": str(output_path),
                "validation_sequences": len(validation),
                "test_sequences": len(test),
                "validation_object_counts": manifest["validation_object_counts"],
                "test_object_counts": manifest["test_object_counts"],
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
