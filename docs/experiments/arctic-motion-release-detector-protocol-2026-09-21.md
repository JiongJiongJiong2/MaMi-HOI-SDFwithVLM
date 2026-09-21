# ARCTIC Motion-Aware Release Detector Protocol

Date: 2026-09-21

Status: frozen before execution

## Goal

Test whether pre-release hand-root and object motion improves the
instant-release role-switch detector over the distance-only model.

The trigger and label definitions are unchanged:

```text
15-frame stable contact
first raw non-contact frame
no future confirmation
```

## Features

The existing 30-frame window contains ten distance/contact features per
frame. Six pre-release motion features are added:

```text
outgoing hand-root speed, mm/frame
other hand-root speed, mm/frame
outgoing-root minus object-translation speed, mm/frame
other-root minus object-translation speed, mm/frame
object translation speed, mm/frame
object angular speed, degrees/frame
```

Velocities use first differences with the first frame zero-filled.
Velocity and angular features are clipped to limit outliers.

No future frame is used.

## Frozen Comparison

The distance-only instant-release baseline is:

```text
boosting train OOF AUPRC: 0.2611
boosting val AUPRC:       0.6484
boosting val F1:          0.6154
```

## Models

The same logistic and HistGradientBoosting configurations and
participant-grouped four-fold threshold selection are reused.

## Gate

The motion-aware pilot passes only if:

1. train and val Tier A trigger recall remain at least 0.95;
2. motion-boosting val AUPRC is at least the frozen value `0.6484`;
3. motion-boosting val F1 is at least the frozen value `0.6154`;
4. motion-boosting train OOF AUPRC exceeds the frozen value `0.2611`.

Official test remains unavailable. Passing establishes a pilot
improvement on accessible participants only.
