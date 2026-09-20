#!/usr/bin/env python3
"""Build a compact contact-memory manifest from EPIC-Contact pickles."""

from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import pickle
import sys
from collections import Counter, deque
from pathlib import Path


CONTACT_THRESHOLD_M = 3e-3


class StopStream(Exception):
    pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-pickle", type=Path)
    parser.add_argument("--test-pickle", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--contact-threshold-m", type=float, default=3e-3)
    parser.add_argument("--max-gap-frames", type=int, default=1)
    parser.add_argument("--max-samples-per-split", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=5000)
    parser.add_argument("--trim-memo", action="store_true")
    parser.add_argument("--memo-window", type=int, default=1000)
    return parser.parse_args()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_identity(path):
    path = Path(path)
    stat = path.stat()
    return {
        "path": str(path),
        "size": int(stat.st_size),
        "sha256": sha256_file(path),
    }


def scalar(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return float(value.reshape(-1)[0])


def contact_state(distance, valid, threshold_m):
    if not bool(scalar(valid)):
        return False, None
    if hasattr(distance, "detach"):
        distance = distance.detach().cpu().numpy()
    values = distance.reshape(-1)
    if values.size == 0:
        return False, None
    minimum = float(values.min())
    return minimum <= threshold_m, minimum


def parse_key(key):
    prefix = str(key).split("_frame_")[0]
    parts = prefix.split("_")
    if len(parts) < 6:
        raise ValueError(f"unexpected EPIC key: {key}")
    return {
        "video_id": "_".join(parts[:2]),
        "clip_id": "_".join(parts[:4]),
        "hand": parts[4],
        "object_name": "_".join(parts[5:]),
    }


def load_pure_unpickler():
    import pickle as pickle_module

    source_path = Path(pickle_module.__file__)
    source = source_path.read_text(encoding="utf-8")
    cutoff = source.rfind("\ntry:\n    from _pickle import")
    if cutoff < 0:
        raise RuntimeError("could not isolate pure-Python pickle source")
    namespace = {
        "__name__": "_epic_pure_pickle",
        "__file__": str(source_path),
    }
    exec(
        compile(source[:cutoff], str(source_path), "exec"),
        namespace,
    )
    return namespace["_Unpickler"]


PureUnpickler = load_pure_unpickler()


def large_value_bytes(value, seen=None):
    if seen is None:
        seen = set()
    value_id = id(value)
    if value_id in seen:
        return 0
    seen.add(value_id)
    module = type(value).__module__.split(".")[0]
    if module == "numpy":
        return int(getattr(value, "nbytes", 0))
    if module == "torch":
        return int(
            getattr(value, "numel", lambda: 0)()
            * getattr(value, "element_size", lambda: 1)()
        )
    if isinstance(value, dict):
        return max(
            [
                large_value_bytes(key, seen)
                for key in value
            ] + [
                large_value_bytes(item, seen)
                for item in value.values()
            ] + [0]
        )
    if isinstance(value, (list, tuple)):
        return max(
            [large_value_bytes(item, seen) for item in value] + [0]
        )
    return 0


def contains_large_value(value, threshold=4096):
    return large_value_bytes(value) > threshold


class StreamingTopLevelUnpickler(PureUnpickler):
    dispatch = PureUnpickler.dispatch.copy()

    def __init__(
        self,
        file,
        on_sample,
        trim_memo=False,
        memo_window=1000,
    ):
        super().__init__(file)
        self.on_sample = on_sample
        self.trim_memo = trim_memo
        self.memo_window = memo_window
        self.top_level_dict = None
        self.sample_count = 0
        self.current_memo_start = 0
        self.memo_retirement = deque()

    def _append(self, value):
        self.stack.append(value)
        if (
            len(self.metastack) == 1
            and len(self.stack) == 1
            and isinstance(value, str)
        ):
            self.current_memo_start = max(self.memo, default=-1) + 1

    def _consume_completed_top_pair(self):
        if (
            len(self.metastack) == 1
            and len(self.stack) >= 2
            and isinstance(self.stack[-2], str)
        ):
            key = self.stack[-2]
            sample = self.stack[-1]
            self.on_sample(key, sample)
            self.sample_count += 1
            if self.top_level_dict is not None:
                self.top_level_dict[key] = None
            del self.stack[-2:]
            self._trim_memo(sample)

    def load_mark(self):
        super().load_mark()
        self.append = self._append

    def pop_mark(self):
        items = super().pop_mark()
        self.append = self._append
        return items

    def load_setitems(self):
        super().load_setitems()
        self._consume_completed_top_pair()

    def load_dict(self):
        super().load_dict()
        self._consume_completed_top_pair()

    def load_empty_dictionary(self):
        super().load_empty_dictionary()
        if self.top_level_dict is None:
            self.top_level_dict = self.stack[-1]

    def _trim_memo(self, sample):
        if not self.trim_memo:
            return
        protected = {
            id(sample[key])
            for key in ("left_valid", "right_valid", "frame_num")
            if key in sample
        }
        # Some EPIC pickle streams reuse memoized objects across top-level
        # records. Replace large values with placeholders instead of deleting
        # their indices so a later BINGET cannot fail.
        for memo_key in range(
            self.current_memo_start,
            max(self.memo, default=-1) + 1,
        ):
            if memo_key not in self.memo:
                continue
            value = self.memo[memo_key]
            if value is self.top_level_dict or value is None:
                continue
            if id(value) in protected:
                continue
            module = type(value).__module__.split(".")[0]
            if (
                value is sample
                or module in {"torch", "numpy"}
                or contains_large_value(value)
            ):
                self.memo_retirement.append((
                    memo_key,
                    self.sample_count,
                ))
        cutoff = self.sample_count - self.memo_window
        while (
            self.memo_retirement
            and self.memo_retirement[0][1] <= cutoff
        ):
            memo_key, _ = self.memo_retirement.popleft()
            if memo_key in self.memo:
                self.memo[memo_key] = None


StreamingTopLevelUnpickler.dispatch[
    pickle.MARK[0]
] = StreamingTopLevelUnpickler.load_mark
StreamingTopLevelUnpickler.dispatch[
    pickle.EMPTY_DICT[0]
] = StreamingTopLevelUnpickler.load_empty_dictionary
StreamingTopLevelUnpickler.dispatch[
    pickle.SETITEMS[0]
] = StreamingTopLevelUnpickler.load_setitems
StreamingTopLevelUnpickler.dispatch[
    pickle.DICT[0]
] = StreamingTopLevelUnpickler.load_dict


class ManifestAccumulator:
    def __init__(self, contact_threshold_m, max_gap_frames):
        self.contact_threshold_m = contact_threshold_m
        self.max_gap_frames = max_gap_frames
        self.frames = {}
        self.hand_records = {}
        self.counts = Counter()

    def add(self, split, key, sample):
        metadata = parse_key(key)
        frame = int(scalar(sample["frame_num"]))
        left_valid = bool(scalar(sample["left_valid"]))
        right_valid = bool(scalar(sample["right_valid"]))
        left_contact, left_min = contact_state(
            sample["dist.lo"],
            sample["left_valid"],
            self.contact_threshold_m,
        )
        right_contact, right_min = contact_state(
            sample["dist.ro"],
            sample["right_valid"],
            self.contact_threshold_m,
        )

        frame_key = (split, metadata["video_id"], frame)
        row = self.frames.setdefault(frame_key, {
            "split": split,
            "video_id": metadata["video_id"],
            "frame": frame,
            "left": {
                "valid": False,
                "contact": False,
                "min_distance_m": None,
                "clip_id": None,
                "object_name": None,
            },
            "right": {
                "valid": False,
                "contact": False,
                "min_distance_m": None,
                "clip_id": None,
                "object_name": None,
            },
        })
        hand = metadata["hand"]
        hand_row = row[hand]
        if hand == "left":
            hand_row["valid"] = left_valid
            hand_row["contact"] = left_contact
            hand_row["min_distance_m"] = left_min
        else:
            hand_row["valid"] = right_valid
            hand_row["contact"] = right_contact
            hand_row["min_distance_m"] = right_min
        hand_row["clip_id"] = metadata["clip_id"]
        hand_row["object_name"] = metadata["object_name"]

        hand_key = (
            split,
            metadata["clip_id"],
            hand,
            metadata["object_name"],
        )
        self.hand_records.setdefault(hand_key, []).append({
            "frame": frame,
            "valid": left_valid if hand == "left" else right_valid,
            "contact": left_contact if hand == "left" else right_contact,
            "min_distance_m": left_min if hand == "left" else right_min,
        })
        self.counts[f"{split}_records"] += 1

    def finalize_frames(self):
        rows = []
        for row in sorted(
            self.frames.values(),
            key=lambda item: (
                item["split"],
                item["video_id"],
                item["frame"],
            ),
        ):
            left_object = row["left"]["object_name"]
            right_object = row["right"]["object_name"]
            same_object = bool(
                row["left"]["valid"]
                and row["right"]["valid"]
                and left_object == right_object
            )
            output = dict(row)
            output["same_object_bimanual"] = same_object
            output["same_object_both_contact"] = bool(
                same_object
                and row["left"]["contact"]
                and row["right"]["contact"]
            )
            rows.append(output)
        return rows

    def finalize_episodes(self):
        episodes = []
        for key, raw_rows in self.hand_records.items():
            split, clip_id, hand, object_name = key
            rows = sorted(raw_rows, key=lambda row: row["frame"])
            segment = []
            for row in rows:
                if (
                    segment
                    and row["frame"] - segment[-1]["frame"]
                    > self.max_gap_frames
                ):
                    episodes.extend(self._episodes_from_segment(
                        split,
                        clip_id,
                        hand,
                        object_name,
                        segment,
                    ))
                    segment = []
                segment.append(row)
            if segment:
                episodes.extend(self._episodes_from_segment(
                    split,
                    clip_id,
                    hand,
                    object_name,
                    segment,
                ))
        return sorted(
            episodes,
            key=lambda row: (
                row["split"],
                row["video_id"],
                row["clip_id"],
                row["hand"],
                row["onset_frame"],
            ),
        )

    def _episodes_from_segment(
        self,
        split,
        clip_id,
        hand,
        object_name,
        segment,
    ):
        episodes = []
        current = []
        for index, row in enumerate(segment):
            if row["contact"]:
                current.append(row)
                continue
            if current:
                episodes.append(self._episode_row(
                    split,
                    clip_id,
                    hand,
                    object_name,
                    current,
                    onset_observed=(
                        current[0] is not segment[0]
                        and not segment[
                            segment.index(current[0]) - 1
                        ]["contact"]
                    ),
                    release_observed=True,
                ))
                current = []
        if current:
            episodes.append(self._episode_row(
                split,
                clip_id,
                hand,
                object_name,
                current,
                onset_observed=(
                    current[0] is not segment[0]
                    and not segment[
                        segment.index(current[0]) - 1
                    ]["contact"]
                ),
                release_observed=False,
            ))
        return episodes

    def _episode_row(
        self,
        split,
        clip_id,
        hand,
        object_name,
        rows,
        onset_observed,
        release_observed,
    ):
        gaps = [
            right["frame"] - left["frame"]
            for left, right in zip(rows, rows[1:])
        ]
        return {
            "split": split,
            "video_id": clip_id.rsplit("_", 2)[0],
            "clip_id": clip_id,
            "hand": hand,
            "object_name": object_name,
            "onset_frame": rows[0]["frame"],
            "last_contact_frame": rows[-1]["frame"],
            "contact_frame_count": len(rows),
            "span": rows[-1]["frame"] - rows[0]["frame"],
            "max_gap": max(gaps) if gaps else 0,
            "onset_observed": bool(onset_observed),
            "release_observed": bool(release_observed),
            "mean_distance_m": float(
                sum(row["min_distance_m"] for row in rows) / len(rows)
            ),
        }


def write_jsonl_gzip(path, rows):
    with gzip.open(path, "wt", encoding="utf-8", compresslevel=6) as handle:
        for row in rows:
            handle.write(json.dumps(row, sort_keys=True) + "\n")


def stream_pickle(
    path,
    accumulator,
    split,
    max_samples,
    progress_every,
    trim_memo,
    memo_window,
):
    processed = 0

    def on_sample(key, sample):
        nonlocal processed
        if max_samples and processed >= max_samples:
            raise StopStream
        accumulator.add(split, key, sample)
        processed += 1
        if progress_every and processed % progress_every == 0:
            print(
                f"{split}: processed {processed}",
                flush=True,
            )

    with Path(path).open("rb") as handle:
        unpickler = StreamingTopLevelUnpickler(
            handle,
            on_sample,
            trim_memo=trim_memo,
            memo_window=memo_window,
        )
        try:
            unpickler.load()
        except StopStream:
            pass
    return processed


def main():
    args = parse_args()
    if not args.train_pickle and not args.test_pickle:
        raise ValueError("at least one pickle is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    accumulator = ManifestAccumulator(
        args.contact_threshold_m,
        args.max_gap_frames,
    )
    sources = {}
    for split, path in (
        ("train", args.train_pickle),
        ("test", args.test_pickle),
    ):
        if path is None:
            continue
        count = stream_pickle(
            path,
            accumulator,
            split,
            args.max_samples_per_split,
            args.progress_every,
            args.trim_memo,
            args.memo_window,
        )
        sources[split] = {
            **file_identity(path),
            "records_processed": count,
        }

    frames = accumulator.finalize_frames()
    episodes = accumulator.finalize_episodes()
    frame_path = args.output_dir / "epic_contact_frames_v1.jsonl.gz"
    episode_path = args.output_dir / "epic_contact_episodes_v1.jsonl.gz"
    write_jsonl_gzip(frame_path, frames)
    write_jsonl_gzip(episode_path, episodes)

    episode_counts = Counter(
        (
            row["split"],
            row["onset_observed"],
            row["release_observed"],
        )
        for row in episodes
    )
    same_object_bimanual = [
        row for row in frames if row["same_object_bimanual"]
    ]
    summary = {
        "contact_threshold_m": args.contact_threshold_m,
        "max_gap_frames": args.max_gap_frames,
        "trim_memo": args.trim_memo,
        "memo_window": args.memo_window,
        "sources": sources,
        "counts": {
            "records": dict(accumulator.counts),
            "merged_frames": len(frames),
            "episodes": len(episodes),
            "same_object_bimanual_frames": len(same_object_bimanual),
            "same_object_both_contact_frames": int(sum(
                row["same_object_both_contact"]
                for row in same_object_bimanual
            )),
        },
        "episode_observation_counts": {
            f"{split}:onset={onset}:release={release}": count
            for (split, onset, release), count in sorted(
                episode_counts.items()
            )
        },
        "outputs": {
            "frames": {
                **file_identity(frame_path),
                "rows": len(frames),
            },
            "episodes": {
                **file_identity(episode_path),
                "rows": len(episodes),
            },
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
