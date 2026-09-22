# EPIC-Contact Lifecycle Memory Protocol

Date: 2026-09-23

Status: frozen before full result analysis

## Goal

Test whether lifecycle-aware material-point memory improves downstream
correction of perturbed EPIC hand trajectories while preserving stable contact
and avoiding post-release attraction.

This is an independent mechanism experiment. It does not depend on A, does not
modify MaMi, and is not a world-model training run.

## Data

Raw inputs:

```text
/root/autodl-tmp/epic_contact_data/training_pkls/
  epic_contact_v3_01-06-26_parallel_train_20260611.pkl
  epic_contact_v3_01-06-26_parallel_test_20260611.pkl
```

Strict split:

```text
/root/autodl-tmp/epic_contact_strict_lifecycle_v1_20260920/
  epic_contact_strict_split_v1.json
```

Primary contact threshold is `1 mm`. The frame and episode datasets are
definitional distance labels, not independent physical release annotation.
Every result must report the `0.5/1.0/1.5/2.0/3.0 mm` sensitivity curve.

The strict test split was already viewed by the EPIC event baseline. It is a
reused locked confirmation, not a pristine final test.

## Correspondence Manifest

The builder streams the raw pickle into compact shards. Each valid
`(split, participant, video, clip, frame, hand)` stores:

- strict contact and minimum hand-object distance;
- MANO pose, beta, translation, and full hand vertices;
- canonical object vertices, faces, diameter, rotation, and translation;
- best and Top-4 hand/object vertex pairs with distances and XYZ positions.

Object topology must be constant within a clip. Memory never crosses a clip,
an object instance, or a frame gap larger than one.

## Reference Memory

New anchors are created only from strict-contact pairs at `1 mm`. Existing
anchors follow one of:

```text
HOLD
UPDATE
CLOSE
UNKNOWN
```

Reference thresholds are selected only on the 25 train participants with
five-fold participant cross-validation. The fixed grid is:

```text
hold ratio:       0.005, 0.01, 0.02, 0.04 * object diameter
update ratio:     0.005, 0.01, 0.02, 0.04 * object diameter
update persistence: 1, 3, 5 frames
close persistence:  1, 3, 5 frames
```

The selected configuration minimizes mean normalized hold error plus update
and false-close penalties. Object-held-out folds exclude the held-out object
class during reference-threshold calibration.

## Sequence Model

```text
architecture: causal GRU
history: 16
layers: 2
hidden size: 128
dropout: 0.1
object embedding: 8
hand embedding: 2
previous-action embedding: 4
optimizer: AdamW
learning rate: 1e-3
weight decay: 1e-4
batch size: 512
epochs: 50 maximum
patience: 7
seed: 20260923
```

The model never receives a bidirectional or future feature. Checkpoints store
the reference configuration, class mapping, model state, and metrics.

## Perturbation And Correction

All policies see the same deterministic low-pass perturbation at:

```text
2 mm, 5 mm, 10 mm
```

The correction freezes the global pose and `hand_mTc`, optimizing finger pose
and root translation:

```text
iterations: 100
learning rate: 0.005
pose cap: min(1.0, 2x observed perturbation magnitude)
translation cap: min(20 mm, 2x observed perturbation magnitude)
contact anchor weight: 10
```

Policy arms:

```text
no correction
ordinary smoothing
per-frame nearest
sticky
hysteresis
oracle
learned
```

## Metrics

Primary:

- end-to-end MANO vertex error against GT;
- hold anchor error;
- post-release attraction;
- false-break rate;
- switch delay;
- acceleration and jerk;
- participant-clustered paired bootstrap intervals.

Event metrics from the sequence model are reported separately and do not
replace the geometric correction endpoint.

## Gate

The oracle must improve both end-to-end correction error and post-release
attraction on dev; otherwise B stops without a larger model.

The learned memory passes only if:

1. test vertex error beats hysteresis and sticky with paired lower bounds
   above zero;
2. post-release attraction beats sticky and per-frame nearest;
3. hold drift and switch delay do not regress by more than 5%;
4. release AUPRC and close F1 do not fall below the frozen logistic baseline;
5. acceleration and jerk do not exceed hysteresis by more than 5%;
6. object-held-out direction agrees in at least three of `bowl`, `plate`,
   `pan`, and `bottle`.

Failure is recorded as NO-GO without changing the test threshold or the
held-out labels.
