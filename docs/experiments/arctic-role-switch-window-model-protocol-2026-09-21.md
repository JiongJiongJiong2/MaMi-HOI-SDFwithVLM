# ARCTIC Role-Switch Window Model Protocol

Date: 2026-09-21

Status: frozen before model training

## Goal

Test whether information available before an outgoing release can
distinguish a clean Tier A role-switch candidate from a raw transition
that fails the quality rules.

This is a candidate-quality prediction task. It is not frame-level
handover detection and cannot establish a human handover label.

## Inputs

```text
tier_a_candidates.jsonl.gz
role_switch_candidates.jsonl.gz
distance_trajectories.npz
```

Primary threshold: `3 mm`.

Every raw candidate at `3 mm` becomes one sample:

```text
Tier A:     label 1
other raw:  label 0
```

## Observation Window

```text
history length: 30 frames
window end:     outgoing release, exclusive
```

No frame at or after the outgoing release is used as model input.

Per-frame features:

```text
right/left hand-object distance, capped at 20 mm
right/left 1-frame distance delta
right/left 5-frame distance delta
right/left contact flags at 3 mm
right/left contact flags at 10 mm
```

The window is flattened for both model arms.

## Splits

The official accessible participant split is mandatory:

```text
train: 8 subjects
val:   s05
```

Out-of-fold predictions use four `GroupKFold` folds over the eight
training participants. The hard F1 threshold is selected on pooled
out-of-fold train predictions, never on val.

Official test remains unavailable.

## Models

### Logistic

```text
StandardScaler
LogisticRegression(C=1.0, class_weight=balanced, solver=liblinear)
```

### Nonlinear

```text
HistGradientBoostingClassifier
max_iter=300, learning_rate=0.05, max_leaf_nodes=15
min_samples_leaf=20, l2_regularization=1.0
positive sample weight=sqrt(negative/positive)
```

There is no hyperparameter search.

## Metrics and Gate

Report val AUROC, AUPRC, F1, precision, and recall. Trivial references
are class prevalence and all-positive F1.

The pilot gate passes only if the best arm:

1. exceeds class prevalence in val AUPRC;
2. exceeds all-positive F1;
3. improves train out-of-fold AUPRC over the same-arm null.

The official-test condition remains unavailable. A pass produces a
pilot model result only.
