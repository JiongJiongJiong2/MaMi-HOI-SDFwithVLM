# EPIC-Contact Strict Nonlinear Event Protocol

Date: 2026-09-20

Status: frozen before training

## Goal

Test whether a nonlinear classifier improves strict-contact event
prediction when it receives exactly the same inputs and uses exactly the
same participant-disjoint split as the learned logistic baseline.

This is a model-capacity experiment. It does not add image, hand-pose,
object-geometry, or future-frame information.

## Frozen Baseline

```text
commit:
8b52863

protocol:
docs/experiments/epic-contact-event-baseline-protocol-2026-09-20.md

result:
docs/experiments/epic-contact-event-baseline-2026-09-20.md

input:
docs/experiments/epic_contact_strict_frames_v1.jsonl.gz
SHA-256
d202ebbdd3c2d907eb0d607794cd6d4cfa7ffdc9f61796628d7a8f26abaa9e36
```

The nonlinear model consumes the baseline feature vector without
adding or removing features.

## Model

One independent `HistGradientBoostingClassifier` is fitted per target:

```text
next_contact
onset
release
```

Hyperparameters are fixed before training:

```text
max_iter:              300
learning_rate:        0.05
max_leaf_nodes:        15
min_samples_leaf:      20
l2_regularization:     1.0
max_bins:              128
early_stopping:        false
positive weighting:    sqrt(negative_count / positive_count)
```

There is no hyperparameter search. Per-target random seeds are the
baseline seed plus the target offset.

## Evaluation

The baseline logistic and two trivial arms are recomputed from the same
samples in the same run. Hard F1 thresholds are selected on dev per
model and target, then frozen for test.

Metrics remain:

```text
next_contact: AUROC, AUPRC, F1
onset:        AUPRC, F1
release:      AUPRC, F1
```

Paired participant-clustered bootstrap uses 2,000 draws and reports
nonlinear-minus-logistic differences with 95% intervals.

## Gate

The nonlinear route continues only if all conditions hold:

1. release AUPRC exceeds learned logistic;
2. the release paired AUPRC lower confidence bound is greater than zero;
3. onset AUPRC is no more than 0.01 below learned logistic;
4. next-contact AUPRC is no more than 0.01 below learned logistic;
5. onset and release F1 are each no more than 0.02 below learned
   logistic.

Failure does not invalidate the strict event baseline. It means that
extra nonlinearity is not justified by this feature set.

Any result continues to require the `0.5-3 mm` threshold sensitivity
curve before a final strict-route decision.

## Boundary

This experiment cannot support handover, receiver-role switching, or
cross-hand transfer claims. The C handover route remains NO-GO.
