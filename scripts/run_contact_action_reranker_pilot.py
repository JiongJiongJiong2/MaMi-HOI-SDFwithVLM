#!/usr/bin/env python3
"""Cheap A/B/C/D pilot for action-chunk reranking.

A is the frozen geometry scorer. B is a pairwise ranker. C is B plus a
supervised contrastive auxiliary. D is an energy regressor without
contrastive learning. The world model and candidate set remain frozen.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn

from manip.world_model.contact_action.features import NORMAL_SLICE
from manip.world_model.contact_action.model import ContactActionTransition
from scripts.run_contact_action_chunk_correction import (
    first_onset_window,
    make_action_candidates,
)
from scripts.select_mami_candidates_with_world_model import (
    candidate_features,
    candidate_files,
)


HAND_NAMES = ("left", "right")
SCALAR_FEATURES = (
    "candidate_frame_contact_fraction",
    "candidate_vertex_contact_fraction",
    "candidate_min_distance",
    "candidate_mean_penetration_distance",
    "candidate_clearance_abs",
    "candidate_contact_probability",
    "candidate_action_magnitude",
    "candidate_action_smoothness",
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate_roots", nargs="+", required=True)
    parser.add_argument("--metric_jsons", nargs="+", required=True)
    parser.add_argument("--world_model_checkpoint", required=True)
    parser.add_argument("--data_root_folder", required=True)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--dataset_cache", default="")
    parser.add_argument("--candidate_seed", type=int, default=1)
    parser.add_argument("--history", type=int, default=4)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--alpha", type=float, default=0.002)
    parser.add_argument("--num_random", type=int, default=8)
    parser.add_argument(
        "--proxy_contact_threshold_norm",
        type=float,
        default=0.02,
    )
    parser.add_argument("--penetration_weight", type=float, default=20.0)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seeds", type=int, nargs="+", default=(0, 1, 2))
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--hidden_size", type=int, default=128)
    parser.add_argument("--embedding_size", type=int, default=64)
    parser.add_argument("--batch_events", type=int, default=16)
    parser.add_argument("--learning_rate", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--contrastive_weight", type=float, default=0.2)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--positive_top_k", type=int, default=3)
    parser.add_argument("--bootstrap_samples", type=int, default=5000)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


class CandidateRanker(nn.Module):
    """Shared-capacity encoder for B, C and D."""

    def __init__(self, input_size, hidden_size, embedding_size):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.score_head = nn.Linear(hidden_size, 1)
        self.contrast_head = nn.Sequential(
            nn.Linear(hidden_size, embedding_size),
            nn.SiLU(),
            nn.Linear(embedding_size, embedding_size),
        )

    def forward(self, features):
        hidden = self.encoder(features)
        score = self.score_head(hidden).squeeze(-1)
        embedding = F.normalize(self.contrast_head(hidden), dim=-1)
        return score, embedding


def load_metric_events(paths):
    events = {}
    for path in paths:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        for event in payload["events"]:
            key = (event["sequence"], event["hand"])
            if key in events:
                raise ValueError(f"duplicate event: {key}")
            events[key] = event
    return events


def build_dataset(args):
    cache_path = Path(args.dataset_cache) if args.dataset_cache else None
    if cache_path is not None and cache_path.exists():
        return dict(np.load(cache_path, allow_pickle=False))

    metric_events = load_metric_events(args.metric_jsons)
    mapping = {}
    for root in args.candidate_roots:
        candidate_dir = (
            Path(root) / f"candidate_seed_{args.candidate_seed}"
        )
        mapping.update(candidate_files(candidate_dir))

    checkpoint = torch.load(
        args.world_model_checkpoint,
        map_location=args.device,
    )
    model = ContactActionTransition(
        hidden_size=checkpoint["args"]["hidden_size"],
        residual_scale=checkpoint["args"].get("residual_scale", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(args.device)
    model.eval()

    data_root = Path(args.data_root_folder)
    rng = np.random.default_rng(args.seed if hasattr(args, "seed") else 1)
    object_scale_cache = {}
    object_sdf_cache = {}
    rows = []

    for sequence_name, candidate_path in sorted(mapping.items()):
        sequence_events = [
            event
            for (sequence, _), event in metric_events.items()
            if sequence == sequence_name
        ]
        if not sequence_events:
            continue
        candidate = np.load(candidate_path, allow_pickle=True)
        features = candidate_features(
            candidate,
            data_root,
            object_scale_cache,
            object_sdf_cache,
            contact_threshold_norm=args.proxy_contact_threshold_norm,
            history=args.history,
            horizon=args.horizon,
            stride=args.stride,
        )
        ground_truth = np.load(
            data_root
            / "contact_labels_w_semantics_npy_files"
            / f"{sequence_name}.npy"
        )[:, :2]

        for hand_index, hand_name in enumerate(HAND_NAMES):
            window_index = first_onset_window(
                features["windows"],
                ground_truth,
                hand_index,
            )
            if window_index is None:
                continue
            normal = features["windows"]["state_history"][
                window_index, -1, NORMAL_SLICE
            ].reshape(2, 3)
            candidates = make_action_candidates(
                features["windows"]["future_actions"][window_index],
                normal,
                hand_index,
                args.alpha,
                args.num_random,
                rng,
            )
            metric_event = metric_events.get((sequence_name, hand_name))
            if metric_event is None:
                continue
            current_index = int(
                features["windows"]["anchor_indices"][window_index]
            )
            if current_index != int(metric_event["current_index"]):
                raise ValueError(
                    "candidate/event mismatch for "
                    f"{sequence_name}/{hand_name}: "
                    f"{current_index} vs {metric_event['current_index']}"
                )
            candidate_actions_np = np.stack(
                [action for _, action in candidates]
            ).astype(np.float32)
            state_history = features["windows"]["state_history"][
                window_index
            ].astype(np.float32)
            action_history = features["windows"]["action_history"][
                window_index
            ].astype(np.float32)
            candidate_actions = torch.from_numpy(
                candidate_actions_np
            ).to(args.device)
            with torch.no_grad():
                output = model.rollout(
                    torch.from_numpy(state_history)[None]
                    .repeat(len(candidates), 1, 1)
                    .to(args.device),
                    torch.from_numpy(action_history)[None]
                    .repeat(len(candidates), 1, 1)
                    .to(args.device),
                    candidate_actions,
                    teacher_states=None,
                    teacher_forcing_ratio=0.0,
                )
            predicted_states = output["states"].detach().cpu().numpy()
            scalar_features = np.stack(
                [
                    np.asarray(metric_event[key], dtype=np.float32)
                    for key in SCALAR_FEATURES
                ],
                axis=1,
            )
            rows.append(
                {
                    "sequence": sequence_name,
                    "object_name": str(candidate["object_name"]),
                    "hand": hand_name,
                    "current_index": current_index,
                    "state_history": state_history,
                    "action_history": action_history,
                    "candidate_actions": candidate_actions_np,
                    "predicted_states": predicted_states,
                    "scalar_features": scalar_features,
                    "candidate_f1": np.asarray(
                        metric_event["candidate_f1"],
                        dtype=np.float32,
                    ),
                    "candidate_mean_penetration_distance": np.asarray(
                        metric_event[
                            "candidate_mean_penetration_distance"
                        ],
                        dtype=np.float32,
                    ),
                    "candidate_frame_contact_fraction": np.asarray(
                        metric_event["candidate_frame_contact_fraction"],
                        dtype=np.float32,
                    ),
                    "selection_score": (
                        np.asarray(
                            metric_event[
                                "candidate_frame_contact_fraction"
                            ],
                            dtype=np.float32,
                        )
                        - args.penetration_weight
                        * np.asarray(
                            metric_event[
                                "candidate_mean_penetration_distance"
                            ],
                            dtype=np.float32,
                        )
                    ),
                }
            )

    if not rows:
        raise RuntimeError("no candidate events were exported")
    dataset = {
        "sequences": np.asarray([row["sequence"] for row in rows]),
        "object_names": np.asarray(
            [row["object_name"] for row in rows]
        ),
        "hands": np.asarray([row["hand"] for row in rows]),
        "current_indices": np.asarray(
            [row["current_index"] for row in rows],
            dtype=np.int64,
        ),
        "state_history": np.stack(
            [row["state_history"] for row in rows]
        ),
        "action_history": np.stack(
            [row["action_history"] for row in rows]
        ),
        "candidate_actions": np.stack(
            [row["candidate_actions"] for row in rows]
        ),
        "predicted_states": np.stack(
            [row["predicted_states"] for row in rows]
        ),
        "scalar_features": np.stack(
            [row["scalar_features"] for row in rows]
        ),
        "candidate_f1": np.stack(
            [row["candidate_f1"] for row in rows]
        ),
        "candidate_penetration": np.stack(
            [row["candidate_mean_penetration_distance"] for row in rows]
        ),
        "selection_score": np.stack(
            [row["selection_score"] for row in rows]
        ),
    }
    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(cache_path, **dataset)
    return dataset


def make_fold_ids(groups, folds, seed):
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    rng.shuffle(unique_groups)
    group_to_fold = {
        group: index % folds
        for index, group in enumerate(unique_groups)
    }
    return np.asarray([group_to_fold[group] for group in groups])


def make_feature_tensor(dataset):
    context = np.concatenate(
        [
            dataset["state_history"].reshape(len(dataset["sequences"]), -1),
            dataset["action_history"].reshape(len(dataset["sequences"]), -1),
        ],
        axis=1,
    )[:, None, :]
    context = np.repeat(
        context,
        dataset["candidate_actions"].shape[1],
        axis=1,
    )
    candidate = np.concatenate(
        [
            dataset["candidate_actions"].reshape(
                len(dataset["sequences"]), dataset["candidate_actions"].shape[1], -1
            ),
            dataset["predicted_states"].reshape(
                len(dataset["sequences"]), dataset["predicted_states"].shape[1], -1
            ),
            dataset["scalar_features"],
        ],
        axis=2,
    )
    return np.concatenate([context, candidate], axis=2).astype(np.float32)


def pairwise_loss(scores, utility):
    utility_diff = utility[:, :, None] - utility[:, None, :]
    valid = utility_diff.abs() > 1e-6
    score_diff = scores[:, :, None] - scores[:, None, :]
    loss = F.softplus(-score_diff * torch.sign(utility_diff))
    weight = (utility_diff.abs() / 0.05).clamp(max=1.0).detach()
    return (loss * weight * valid).sum() / valid.sum().clamp_min(1)


def supervised_contrastive_loss(
    embedding,
    utility,
    top_k,
    temperature,
):
    batch_size, candidate_count, _ = embedding.shape
    losses = []
    for batch_index in range(batch_size):
        row_utility = utility[batch_index]
        if torch.unique(row_utility).numel() <= 1:
            continue
        positive_count = min(top_k, candidate_count)
        positive_indices = torch.topk(
            row_utility,
            k=positive_count,
            largest=True,
        ).indices
        positive_mask = torch.zeros(
            candidate_count,
            dtype=torch.bool,
            device=embedding.device,
        )
        positive_mask[positive_indices] = True
        similarity = (
            embedding[batch_index] @ embedding[batch_index].transpose(0, 1)
        ) / temperature
        identity = torch.eye(
            candidate_count,
            dtype=torch.bool,
            device=embedding.device,
        )
        positive_pairs = (
            positive_mask[:, None]
            & positive_mask[None, :]
            & ~identity
        )
        denominator = ~identity
        for anchor in torch.nonzero(
            positive_mask,
            as_tuple=False,
        ).flatten():
            log_denominator = torch.logsumexp(
                similarity[anchor][denominator[anchor]],
                dim=0,
            )
            log_positive = torch.logsumexp(
                similarity[anchor][positive_pairs[anchor]],
                dim=0,
            )
            losses.append(-(log_positive - log_denominator))
    if not losses:
        return embedding.new_zeros(())
    return torch.stack(losses).mean()


def train_one_fold(
    arm,
    train_x,
    train_utility,
    test_x,
    seed,
    args,
):
    torch.manual_seed(seed)
    model = CandidateRanker(
        train_x.shape[-1],
        args.hidden_size,
        args.embedding_size,
    ).to(args.device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    train_x = torch.from_numpy(train_x).to(args.device)
    train_utility = torch.from_numpy(train_utility).to(args.device)
    test_x = torch.from_numpy(test_x).to(args.device)
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)

    for _ in range(args.epochs):
        model.train()
        order = torch.randperm(
            train_x.shape[0],
            generator=generator,
        ).to(args.device)
        for start in range(0, len(order), args.batch_events):
            batch_indices = order[start : start + args.batch_events]
            batch_x = train_x[batch_indices]
            batch_utility = train_utility[batch_indices]
            score, embedding = model(batch_x)
            if arm == "D":
                center = batch_utility.mean(dim=1, keepdim=True)
                scale = batch_utility.std(dim=1, keepdim=True).clamp_min(1e-6)
                target_energy = -(
                    (batch_utility - center) / scale
                )
                loss = F.smooth_l1_loss(score, target_energy)
            else:
                loss = pairwise_loss(score, batch_utility)
                if arm == "C":
                    contrastive = supervised_contrastive_loss(
                        embedding,
                        batch_utility,
                        args.positive_top_k,
                        args.temperature,
                    )
                    loss = loss + args.contrastive_weight * contrastive
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

    model.eval()
    with torch.no_grad():
        score, _ = model(test_x)
        if arm == "D":
            score = -score
    return score.cpu().numpy()


def per_event_metrics(scores, dataset, penetration_weight):
    utility = (
        dataset["candidate_f1"]
        - penetration_weight * dataset["candidate_penetration"]
    )
    selected = np.argmax(scores, axis=1)
    row = np.arange(len(selected))
    top_utility = utility[row, selected]
    top_f1 = dataset["candidate_f1"][row, selected]
    top_penetration = dataset["candidate_penetration"][row, selected]
    utility_diff = utility[:, :, None] - utility[:, None, :]
    score_diff = scores[:, :, None] - scores[:, None, :]
    valid_pairs = np.abs(utility_diff) > 1e-6
    pair_correct = score_diff * np.sign(utility_diff) > 0
    pair_accuracy = (
        (pair_correct & valid_pairs).sum(axis=(1, 2))
        / valid_pairs.sum(axis=(1, 2)).clip(min=1)
    )
    oracle_utility = utility.max(axis=1)
    oracle_index = utility.argmax(axis=1)
    top3 = np.argpartition(scores, -min(3, scores.shape[1]), axis=1)[
        :, -min(3, scores.shape[1]) :
    ]
    top3_recall = np.any(top3 == oracle_index[:, None], axis=1).astype(float)
    return {
        "top1_utility": top_utility,
        "top1_f1": top_f1,
        "top1_penetration": top_penetration,
        "pairwise_accuracy": pair_accuracy,
        "oracle_regret": oracle_utility - top_utility,
        "top3_oracle_recall": top3_recall,
    }


def grouped_bootstrap(values, groups, samples, seed):
    values = np.asarray(values, dtype=np.float64)
    groups = np.asarray(groups)
    unique_groups = np.unique(groups)
    rng = np.random.default_rng(seed)
    group_values = {
        group: values[groups == group]
        for group in unique_groups
    }
    draws = np.empty(samples, dtype=np.float64)
    for draw in range(samples):
        sampled_groups = rng.choice(
            unique_groups,
            size=len(unique_groups),
            replace=True,
        )
        draws[draw] = np.mean(
            [
                group_values[group].mean()
                for group in sampled_groups
            ]
        )
    return [
        float(np.quantile(draws, 0.025)),
        float(np.quantile(draws, 0.975)),
    ]


def summarize_arm(metrics, dataset, args):
    summary = {}
    for key, values in metrics.items():
        summary[key] = float(np.mean(values))
        if key in (
            "top1_utility",
            "top1_f1",
            "top1_penetration",
            "pairwise_accuracy",
            "oracle_regret",
            "top3_oracle_recall",
        ):
            summary[f"{key}_ci95"] = grouped_bootstrap(
                values,
                dataset["sequences"],
                args.bootstrap_samples,
                101,
            )
    return summary


def run_experiment(dataset, args):
    candidate_count = dataset["candidate_actions"].shape[1]
    utility = (
        dataset["candidate_f1"]
        - args.penetration_weight * dataset["candidate_penetration"]
    )
    feature_tensor = make_feature_tensor(dataset)
    arm_predictions = {
        arm: np.empty(
            (len(args.seeds), len(dataset["sequences"]), candidate_count),
            dtype=np.float32,
        )
        for arm in ("B", "C", "D")
    }

    for seed_index, seed in enumerate(args.seeds):
        fold_ids = make_fold_ids(
            dataset["sequences"],
            args.folds,
            seed=1000 + seed,
        )
        for arm in ("B", "C", "D"):
            for fold in range(args.folds):
                train_mask = fold_ids != fold
                test_mask = ~train_mask
                train_x = feature_tensor[train_mask]
                test_x = feature_tensor[test_mask]
                center = train_x.mean(axis=(0, 1), keepdims=True)
                scale = train_x.std(axis=(0, 1), keepdims=True).clip(min=1e-6)
                train_x = (train_x - center) / scale
                test_x = (test_x - center) / scale
                predictions = train_one_fold(
                    arm,
                    train_x,
                    utility[train_mask],
                    test_x,
                    seed=seed * 100 + fold,
                    args=args,
                )
                arm_predictions[arm][seed_index, test_mask] = predictions

    predictions = {"A": dataset["selection_score"][None]}
    for arm in ("B", "C", "D"):
        predictions[arm] = arm_predictions[arm]

    event_metrics = {}
    summary = {}
    for arm, arm_scores in predictions.items():
        if arm == "A":
            arm_event_metrics = per_event_metrics(
                arm_scores[0],
                dataset,
                args.penetration_weight,
            )
        else:
            per_seed = [
                per_event_metrics(
                    seed_scores,
                    dataset,
                    args.penetration_weight,
                )
                for seed_scores in arm_scores
            ]
            arm_event_metrics = {
                key: np.mean([item[key] for item in per_seed], axis=0)
                for key in per_seed[0]
            }
        event_metrics[arm] = arm_event_metrics
        summary[arm] = summarize_arm(arm_event_metrics, dataset, args)

    comparisons = {}
    for left, right in (
        ("B", "A"),
        ("C", "A"),
        ("D", "A"),
        ("C", "B"),
        ("D", "B"),
    ):
        comparison = {}
        for key in event_metrics[left]:
            delta = (
                event_metrics[left][key] - event_metrics[right][key]
            )
            comparison[key] = {
                "mean": float(np.mean(delta)),
                "ci95": grouped_bootstrap(
                    delta,
                    dataset["sequences"],
                    args.bootstrap_samples,
                    202,
                ),
            }
        comparisons[f"{left}-{right}"] = comparison

    c_minus_b_top1 = comparisons["C-B"]["top1_utility"]
    c_minus_b_pair = comparisons["C-B"]["pairwise_accuracy"]
    contrastive_gate = (
        c_minus_b_top1["mean"] > 0
        and c_minus_b_top1["ci95"][0] > 0
        and c_minus_b_pair["mean"] > 0
    )
    return {
        "event_count": len(dataset["sequences"]),
        "sequence_count": len(np.unique(dataset["sequences"])),
        "object_count": len(np.unique(dataset["object_names"])),
        "candidate_count": candidate_count,
        "feature_count": int(feature_tensor.shape[-1]),
        "folds": args.folds,
        "seeds": list(args.seeds),
        "penetration_weight": args.penetration_weight,
        "contrastive_weight": args.contrastive_weight,
        "summary": summary,
        "comparisons": comparisons,
        "contrastive_gate": contrastive_gate,
        "event_metrics": {
            arm: {
                key: values.tolist()
                for key, values in arm_metrics.items()
            }
            for arm, arm_metrics in event_metrics.items()
        },
        "sequences": dataset["sequences"].tolist(),
        "objects": dataset["object_names"].tolist(),
        "hands": dataset["hands"].tolist(),
    }


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = (
        Path(args.dataset_cache)
        if args.dataset_cache
        else output_dir / "candidate_dataset.npz"
    )
    args.dataset_cache = str(cache_path)
    dataset = build_dataset(args)
    result = run_experiment(dataset, args)
    result["dataset_cache"] = str(cache_path)
    result["metric_jsons"] = list(args.metric_jsons)
    result["candidate_roots"] = list(args.candidate_roots)
    result["world_model_checkpoint"] = args.world_model_checkpoint
    result_path = output_dir / "reranker_pilot_result.json"
    result_path.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                key: result[key]
                for key in (
                    "event_count",
                    "sequence_count",
                    "object_count",
                    "feature_count",
                    "contrastive_gate",
                    "summary",
                    "comparisons",
                )
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
