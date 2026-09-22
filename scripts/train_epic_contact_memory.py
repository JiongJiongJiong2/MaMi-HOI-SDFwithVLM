#!/usr/bin/env python3
"""Train a causal lifecycle-memory model on EPIC correspondence shards."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


ACTIONS = ("hold", "update", "close", "unknown")
ACTION_TO_ID = {name: index for index, name in enumerate(ACTIONS)}
OBJECT_CLASSES = (
    "bowl",
    "bottle",
    "can",
    "cup",
    "glass",
    "mug",
    "pan",
    "plate",
    "saucepan",
)
OBJECT_TO_ID = {
    name: index for index, name in enumerate(OBJECT_CLASSES)
}
SUPPORTED_OBJECT_FOLDS = ("bowl", "plate", "pan", "bottle")
CONTACT_THRESHOLD_M = 0.001


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--history", type=int, default=16)
    parser.add_argument("--hidden-size", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--seed", type=int, default=20260923)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--held-out-object", choices=OBJECT_CLASSES)
    parser.add_argument("--reference-config-json", type=Path)
    parser.add_argument("--max-train-sequences", type=int, default=0)
    parser.add_argument(
        "--reference-grid-ratios",
        nargs="+",
        type=float,
        default=(0.005, 0.01, 0.02, 0.04),
    )
    parser.add_argument(
        "--reference-grid-persistence",
        nargs="+",
        type=int,
        default=(1, 3, 5),
    )
    return parser.parse_args()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def load_index(manifest_dir):
    rows = []
    with (manifest_dir / "index.jsonl").open(
        "r",
        encoding="utf-8",
    ) as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if row.get("relative_path"):
                    row["path"] = str(
                        manifest_dir
                        / row["relative_path"].replace("\\", "/")
                    )
                rows.append(row)
    if not rows:
        raise ValueError("manifest index is empty")
    return rows


def causal_window(values, history):
    values = np.asarray(values)
    if values.ndim < 1:
        raise ValueError("values must have a time dimension")
    length = min(len(values), history)
    window = np.zeros(
        (history, *values.shape[1:]),
        dtype=values.dtype,
    )
    window[-length:] = values[-length:]
    return window


def closest_candidate(
    object_indices,
    hand_indices,
    target_object,
    target_hand,
):
    object_indices = np.asarray(object_indices)
    hand_indices = np.asarray(hand_indices)
    object_match = np.flatnonzero(object_indices == target_object)
    if len(object_match):
        return int(object_match[0])
    hand_match = np.flatnonzero(hand_indices == target_hand)
    if len(hand_match):
        return int(hand_match[0])
    return None


def reference_tracks(
    arrays,
    object_name,
    hand,
    hold_ratio,
    update_ratio,
    update_persistence,
    close_persistence,
    history,
    contact_threshold_m=CONTACT_THRESHOLD_M,
):
    contact = np.asarray(arrays["contact"], dtype=bool)
    min_distance = np.asarray(
        arrays["min_distance_m"],
        dtype=np.float64,
    )
    top_hand = np.asarray(
        arrays["top_hand_indices"],
        dtype=np.int64,
    )
    top_object = np.asarray(
        arrays["top_object_indices"],
        dtype=np.int64,
    )
    top_distance = np.asarray(
        arrays["top_distances_m"],
        dtype=np.float64,
    )
    hand_vertices = np.asarray(
        arrays["hand_vertices"],
        dtype=np.float64,
    )
    object_vertices = np.asarray(
        arrays["object_vertices"],
        dtype=np.float64,
    )
    rotations = np.asarray(
        arrays["object_rotation"],
        dtype=np.float64,
    )
    translations = np.asarray(
        arrays["object_translation"],
        dtype=np.float64,
    )
    diameter = float(np.asarray(arrays["object_diameter_m"]))
    if diameter <= 0:
        raise ValueError("object diameter must be positive")
    normalized_hold = hold_ratio * diameter
    normalized_update = update_ratio * diameter

    active = {}
    next_track = 0
    tracks = {}

    for time_index in range(len(contact)):
        frame_candidates = [
            index
            for index, distance in enumerate(top_distance[time_index])
            if np.isfinite(distance)
            and distance <= contact_threshold_m
        ]
        if contact[time_index]:
            for candidate_index in frame_candidates:
                object_index = int(top_object[time_index, candidate_index])
                if object_index in active:
                    continue
                if len(active) >= top_object.shape[1]:
                    break
                track_id = next_track
                next_track += 1
                active[object_index] = track_id
                tracks[track_id] = {
                    "object_index": object_index,
                    "hand_index": int(
                        top_hand[time_index, candidate_index]
                    ),
                    "object_name": object_name,
                    "hand": hand,
                    "history": [],
                    "labels": [],
                    "start_time": time_index,
                    "update_counter": 0,
                    "close_counter": 0,
                }

        for object_index in list(active):
            track_id = active[object_index]
            track = tracks[track_id]
            candidate_index = closest_candidate(
                top_object[time_index],
                top_hand[time_index],
                object_index,
                track["hand_index"],
            )
            if candidate_index is None:
                if contact[time_index]:
                    label = ACTION_TO_ID["unknown"]
                else:
                    track["close_counter"] += 1
                    label = (
                        ACTION_TO_ID["close"]
                        if track["close_counter"] >= close_persistence
                        else ACTION_TO_ID["hold"]
                    )
            else:
                hand_index = int(
                    top_hand[time_index, candidate_index]
                )
                object_world = (
                    object_vertices[object_index]
                    @ rotations[time_index].T
                    + translations[time_index]
                )
                anchor_error = np.linalg.norm(
                    hand_vertices[time_index, hand_index]
                    - object_world
                )
                if not contact[time_index]:
                    track["close_counter"] += 1
                    label = (
                        ACTION_TO_ID["close"]
                        if track["close_counter"] >= close_persistence
                        else ACTION_TO_ID["hold"]
                    )
                elif anchor_error > normalized_update:
                    track["update_counter"] += 1
                    if track["update_counter"] >= update_persistence:
                        label = ACTION_TO_ID["update"]
                    else:
                        label = ACTION_TO_ID["unknown"]
                elif anchor_error <= normalized_hold:
                    track["close_counter"] = 0
                    track["update_counter"] = 0
                    label = ACTION_TO_ID["hold"]
                else:
                    label = ACTION_TO_ID["unknown"]

                if label == ACTION_TO_ID["update"]:
                    old_hand = track["hand_index"]
                    track["hand_index"] = hand_index
                    track["object_index"] = int(
                        top_object[time_index, candidate_index]
                    )
                    active.pop(object_index, None)
                    active[track["object_index"]] = track_id
                    track["update_counter"] = 0

                previous_hand = (
                    track["history"][-1]["hand_xyz"]
                    if track["history"]
                    else hand_vertices[time_index, hand_index]
                )
                previous_object = (
                    track["history"][-1]["object_xyz"]
                    if track["history"]
                    else object_world
                )
                hand_delta = np.linalg.norm(
                    hand_vertices[time_index, hand_index]
                    - previous_hand
                )
                object_delta = np.linalg.norm(
                    object_world - previous_object
                )
                features = np.concatenate([
                    np.asarray([
                        float(contact[time_index]),
                        anchor_error / diameter,
                        float(min_distance[time_index]) / diameter,
                        (time_index - track["start_time"]) / 100.0,
                        hand_delta / diameter,
                        object_delta / diameter,
                    ], dtype=np.float32),
                    np.asarray(
                        top_distance[time_index],
                        dtype=np.float32,
                    ) / diameter,
                ])
                if track["history"]:
                    features[3] = min(features[3], 1.0)
                track["history"].append({
                    "features": features,
                    "label": label,
                    "time_index": time_index,
                    "hand_index": hand_index,
                    "object_index": int(
                        top_object[time_index, candidate_index]
                    ),
                    "hand_xyz": hand_vertices[time_index, hand_index].copy(),
                    "object_xyz": object_world.copy(),
                })
                track["labels"].append(label)

            if label == ACTION_TO_ID["close"]:
                active.pop(object_index, None)
                tracks[track_id]["closed_at"] = time_index

    sequences = []
    for track in tracks.values():
        history_rows = track["history"]
        if len(history_rows) < 2:
            continue
        sequences.append({
            "features": np.stack([
                row["features"] for row in history_rows
            ]),
            "labels": np.asarray(track["labels"], dtype=np.int64),
            "times": np.asarray([
                row["time_index"] for row in history_rows
            ], dtype=np.int64),
            "hand_indices": np.asarray([
                row["hand_index"] for row in history_rows
            ], dtype=np.int64),
            "object_indices": np.asarray([
                row["object_index"] for row in history_rows
            ], dtype=np.int64),
            "object_name": track["object_name"],
            "hand": track["hand"],
            "start_time": track["start_time"],
            "length": len(history_rows),
        })
    return sequences


def reference_cost(sequences):
    hold_error = []
    updates = 0
    false_closes = 0
    frames = 0
    for sequence in sequences:
        labels = sequence["labels"]
        features = sequence["features"]
        frames += len(labels)
        updates += int(np.sum(labels == ACTION_TO_ID["update"]))
        false_closes += int(np.sum(
            (labels == ACTION_TO_ID["close"])
            & (features[:, 0] > 0.5)
        ))
        hold_mask = labels == ACTION_TO_ID["hold"]
        hold_error.extend(features[hold_mask, 1].tolist())
    mean_hold_error = (
        float(np.mean(hold_error)) if hold_error else 1.0
    )
    return (
        mean_hold_error
        + 0.02 * updates / max(frames, 1)
        + 0.10 * false_closes / max(frames, 1)
    )


def load_arrays(row):
    with np.load(row["path"], allow_pickle=False) as archive:
        return {
            key: np.asarray(archive[key])
            for key in archive.files
        }


def calibrate_reference_config(args, rows, cache):
    if args.reference_config_json is not None:
        return json.loads(
            args.reference_config_json.read_text(encoding="utf-8")
        )
    train_rows = [
        row for row in rows if row["split"] == "train"
    ]
    participants = sorted({
        row["participant_id"] for row in train_rows
    })
    if len(participants) < 2:
        raise ValueError("at least two train participants are required")
    folds = [
        participants[index::5]
        for index in range(5)
        if participants[index::5]
    ]
    grid = []
    for hold_ratio in args.reference_grid_ratios:
        for update_ratio in args.reference_grid_ratios:
            if update_ratio < hold_ratio:
                continue
            for update_persistence in args.reference_grid_persistence:
                for close_persistence in args.reference_grid_persistence:
                    grid.append({
                        "hold_ratio": float(hold_ratio),
                        "update_ratio": float(update_ratio),
                        "update_persistence": int(update_persistence),
                        "close_persistence": int(close_persistence),
                    })
    scored = []
    for config in grid:
        fold_costs = []
        for fold in folds:
            fold_participants = set(fold)
            sequences = []
            for row in train_rows:
                if row["participant_id"] not in fold_participants:
                    continue
                arrays = cache.get(row["path"])
                if arrays is None:
                    arrays = load_arrays(row)
                    cache[row["path"]] = arrays
                sequences.extend(reference_tracks(
                    arrays,
                    row["object_name"],
                    row["hand"],
                    history=args.history,
                    **config,
                ))
            fold_costs.append(reference_cost(sequences))
        scored.append((float(np.mean(fold_costs)), config))
    scored.sort(key=lambda item: (item[0], json.dumps(item[1])))
    return {
        "selected": scored[0][1],
        "cost": scored[0][0],
        "fold_count": len(folds),
        "grid_size": len(grid),
    }


def build_problem(args, rows, reference_config, cache):
    selected = reference_config["selected"]
    sequences_by_split = defaultdict(list)
    for row in rows:
        if (
            args.held_out_object is not None
            and row["object_name"] != args.held_out_object
        ):
            continue
        arrays = cache.get(row["path"])
        if arrays is None:
            arrays = load_arrays(row)
            cache[row["path"]] = arrays
        sequences = reference_tracks(
            arrays,
            row["object_name"],
            row["hand"],
            history=args.history,
            **selected,
        )
        sequences_by_split[row["split"]].extend(sequences)

    if args.held_out_object is not None:
        train = []
        val = []
        for row in rows:
            if row["split"] != "train":
                continue
            split = (
                "val"
                if row["object_name"] == args.held_out_object
                else "train"
            )
            arrays = cache.get(row["path"])
            if arrays is None:
                arrays = load_arrays(row)
                cache[row["path"]] = arrays
            sequences = reference_tracks(
                arrays,
                row["object_name"],
                row["hand"],
                history=args.history,
                **selected,
            )
            (val if split == "val" else train).extend(sequences)
        sequences_by_split = defaultdict(list, {
            "train": train,
            "dev": val,
            "test": [],
        })

    feature_dim = None
    datasets = {}
    for split, sequences in sequences_by_split.items():
        features = []
        previous_actions = []
        object_ids = []
        hand_ids = []
        labels = []
        for sequence in sequences:
            sequence_features = sequence["features"]
            sequence_labels = sequence["labels"]
            feature_dim = sequence_features.shape[1]
            previous = np.concatenate([
                np.asarray([ACTION_TO_ID["unknown"]], dtype=np.int64),
                sequence_labels[:-1],
            ])
            for time_index in range(len(sequence_labels)):
                features.append(causal_window(
                    sequence_features[: time_index + 1],
                    args.history,
                ))
                previous_actions.append(causal_window(
                    previous[: time_index + 1],
                    args.history,
                ))
                labels.append(sequence_labels[time_index])
                object_ids.append(OBJECT_TO_ID[sequence["object_name"]])
                hand_ids.append(0 if sequence["hand"] == "left" else 1)
        datasets[split] = {
            "features": np.asarray(features, dtype=np.float32),
            "previous_actions": np.asarray(
                previous_actions,
                dtype=np.int64,
            ),
            "object_ids": np.asarray(object_ids, dtype=np.int64),
            "hand_ids": np.asarray(hand_ids, dtype=np.int64),
            "labels": np.asarray(labels, dtype=np.int64),
        }
    return feature_dim, datasets


def class_weights(labels, classes=len(ACTIONS)):
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    total = counts.sum()
    weights = np.sqrt(total / np.maximum(counts, 1.0) / classes)
    weights = np.clip(weights, 0.25, 5.0)
    return weights.astype(np.float32)


def macro_f1(labels, predictions, classes=len(ACTIONS)):
    scores = []
    for class_id in range(classes):
        true = labels == class_id
        predicted = predictions == class_id
        tp = int(np.sum(true & predicted))
        fp = int(np.sum(~true & predicted))
        fn = int(np.sum(true & ~predicted))
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        scores.append(
            2.0 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
    return float(np.mean(scores)), scores


def build_model(args, feature_dim):
    import torch
    from torch import nn

    class MemoryModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.object_embedding = nn.Embedding(
                len(OBJECT_CLASSES),
                8,
            )
            self.hand_embedding = nn.Embedding(2, 2)
            self.action_embedding = nn.Embedding(len(ACTIONS), 4)
            input_size = feature_dim + 8 + 2 + 4
            self.gru = nn.GRU(
                input_size,
                args.hidden_size,
                num_layers=args.layers,
                dropout=args.dropout if args.layers > 1 else 0.0,
                batch_first=True,
            )
            self.head = nn.Linear(args.hidden_size, len(ACTIONS))

        def forward(
            self,
            features,
            previous_actions,
            object_ids,
            hand_ids,
        ):
            batch, history, _ = features.shape
            object_embedding = self.object_embedding(object_ids)
            hand_embedding = self.hand_embedding(hand_ids)
            action_embedding = self.action_embedding(previous_actions)
            repeated = torch.cat([
                object_embedding[:, None, :].expand(
                    -1,
                    history,
                    -1,
                ),
                hand_embedding[:, None, :].expand(
                    -1,
                    history,
                    -1,
                ),
                action_embedding,
            ], dim=2)
            output, _ = self.gru(torch.cat([
                features,
                repeated,
            ], dim=2))
            return self.head(output[:, -1])

    return MemoryModel()


def tensor_dataset(dataset, device):
    import torch

    return {
        key: torch.as_tensor(value, device=device)
        for key, value in dataset.items()
    }


def evaluate_model(model, dataset, batch_size, device):
    import torch

    model.eval()
    predictions = []
    probabilities = []
    with torch.no_grad():
        for start in range(0, len(dataset["labels"]), batch_size):
            stop = start + batch_size
            logits = model(
                dataset["features"][start:stop],
                dataset["previous_actions"][start:stop],
                dataset["object_ids"][start:stop],
                dataset["hand_ids"][start:stop],
            )
            probabilities.append(
                torch.softmax(logits, dim=1).cpu().numpy()
            )
            predictions.append(logits.argmax(dim=1).cpu().numpy())
    return (
        np.concatenate(predictions),
        np.concatenate(probabilities),
    )


def average_precision_binary(labels, scores):
    labels = np.asarray(labels, dtype=bool)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int(labels.sum())
    if positives == 0:
        return None
    order = np.argsort(-scores, kind="stable")
    sorted_labels = labels[order]
    cumulative = np.cumsum(sorted_labels)
    precision = cumulative / np.arange(
        1,
        len(sorted_labels) + 1,
    )
    return float(precision[sorted_labels].sum() / positives)


def metric_payload(labels, predictions, probabilities):
    macro, per_class = macro_f1(labels, predictions)
    return {
        "macro_f1": macro,
        "per_class_f1": {
            name: per_class[index]
            for index, name in enumerate(ACTIONS)
        },
        "release_auprc": average_precision_binary(
            labels == ACTION_TO_ID["close"],
            probabilities[:, ACTION_TO_ID["close"]],
        ),
        "class_counts": dict(Counter(labels.tolist())),
        "prediction_counts": dict(Counter(predictions.tolist())),
    }


def main():
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(
        (args.manifest_dir / "manifest.json").read_text(
            encoding="utf-8"
        )
    )
    rows = load_index(args.manifest_dir)
    if args.max_train_sequences:
        limited = []
        train_count = 0
        for row in rows:
            if row["split"] == "train":
                if train_count >= args.max_train_sequences:
                    continue
                train_count += 1
            limited.append(row)
        rows = limited
    cache = {}
    calibration_rows = rows
    if args.held_out_object is not None:
        calibration_rows = [
            row for row in rows
            if row["object_name"] != args.held_out_object
        ]
    reference = calibrate_reference_config(
        args,
        calibration_rows,
        cache,
    )
    write_json(
        args.output_dir / "reference_config.json",
        reference,
    )
    feature_dim, datasets = build_problem(
        args,
        rows,
        reference,
        cache,
    )
    if feature_dim is None:
        raise ValueError("no lifecycle histories were found")

    import torch

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args, feature_dim).to(device)
    train = tensor_dataset(datasets["train"], device)
    dev = tensor_dataset(datasets["dev"], device)
    if len(train["labels"]) == 0 or len(dev["labels"]) == 0:
        raise ValueError("train and dev histories are required")
    weights = torch.as_tensor(
        class_weights(datasets["train"]["labels"]),
        device=device,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    loss_function = torch.nn.CrossEntropyLoss(weight=weights)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(args.seed)
    best = None
    stale = 0
    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        order = torch.randperm(
            len(train["labels"]),
            generator=generator,
        ).numpy()
        losses = []
        for start in range(0, len(order), args.batch_size):
            indices = torch.as_tensor(
                order[start : start + args.batch_size],
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            logits = model(
                train["features"][indices],
                train["previous_actions"][indices],
                train["object_ids"][indices],
                train["hand_ids"][indices],
            )
            loss = loss_function(logits, train["labels"][indices])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        predictions, probabilities = evaluate_model(
            model,
            dev,
            args.batch_size,
            device,
        )
        metrics = metric_payload(
            datasets["dev"]["labels"],
            predictions,
            probabilities,
        )
        row = {
            "epoch": epoch,
            "train_loss": float(np.mean(losses)),
            **metrics,
        }
        history.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
        score = metrics["macro_f1"]
        if best is None or score > best["score"]:
            best = {
                "score": score,
                "epoch": epoch,
                "state_dict": {
                    key: value.detach().cpu()
                    for key, value in model.state_dict().items()
                },
                "metrics": metrics,
            }
            stale = 0
        else:
            stale += 1
            if stale >= args.patience:
                break

    model.load_state_dict(best["state_dict"])
    model.to(device)
    all_metrics = {}
    predictions_by_split = {}
    for split in ("train", "dev", "test"):
        if len(datasets[split]["labels"]) == 0:
            all_metrics[split] = None
            predictions_by_split[split] = None
            continue
        split_tensor = tensor_dataset(datasets[split], device)
        predictions, probabilities = evaluate_model(
            model,
            split_tensor,
            args.batch_size,
            device,
        )
        all_metrics[split] = metric_payload(
            datasets[split]["labels"],
            predictions,
            probabilities,
        )
        predictions_by_split[split] = {
            "predictions": predictions,
            "probabilities": probabilities.astype(np.float16),
        }

    checkpoint = {
        "state_dict": model.state_dict(),
        "feature_dim": feature_dim,
        "history": args.history,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "dropout": args.dropout,
        "object_classes": list(OBJECT_CLASSES),
        "actions": list(ACTIONS),
        "reference": reference,
        "manifest": str(args.manifest_dir),
        "manifest_sha256": manifest.get("strict_split_sha256"),
        "held_out_object": args.held_out_object,
        "seed": args.seed,
        "metrics": all_metrics,
    }
    torch.save(checkpoint, args.output_dir / "checkpoint.pt")
    np.savez_compressed(
        args.output_dir / "predictions.npz",
        **{
            f"{split}_{key}": value
            for split, payload in predictions_by_split.items()
            if payload is not None
            for key, value in payload.items()
        },
    )
    summary = {
        "method": (
            "causal GRU over material-point anchor histories with "
            "HOLD/UPDATE/CLOSE/UNKNOWN outputs"
        ),
        "reference": reference,
        "feature_dim": feature_dim,
        "history": args.history,
        "hidden_size": args.hidden_size,
        "layers": args.layers,
        "seed": args.seed,
        "held_out_object": args.held_out_object,
        "best_epoch": best["epoch"],
        "metrics": all_metrics,
        "epoch_history": history,
        "test_note": (
            "strict test split is reused from the EPIC event baseline"
        ),
    }
    write_json(args.output_dir / "metrics.json", summary)
    print(json.dumps({
        "checkpoint": str(args.output_dir / "checkpoint.pt"),
        "best_epoch": best["epoch"],
        "metrics": all_metrics,
    }, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
