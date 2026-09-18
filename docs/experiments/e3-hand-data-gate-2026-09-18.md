# E3 Hand Data Gate

Date: 2026-09-18

Status: E3A blocked; E3B hand-model assets pass; processed-cache hand pose blocked

## Scope

This is the first CPU-only step of the finger-aware WM route. It checks
whether the existing server data and body-model assets can support a
`HandSequenceV1` sidecar. No GPU inference or WM training is involved.

## Results

### Raw finger supervision

No raw BEHAVE, GRAB, ARCTIC, HOT3D, or EPIC-Contact files were found under
the current AutoDL data directories. The exact BEHAVE
`smpl_fit_all.npz` source is therefore unavailable on the server.

Gate result:

```text
E3A raw finger supervision: BLOCKED
```

### Processed MaMi cache

The compact validation sequence cache exposes:

```text
seq_name, betas, gender, trans2joint, rest_offsets, trans,
root_orient, pose_body, obj_scale, obj_trans, obj_rot, obj_com_pos
```

It contains `pose_body` but not `pose_hand`. The current cache therefore
cannot be used to recover articulated finger motion.

Gate result:

```text
E3B processed-cache pose_hand: BLOCKED
```

### Available model assets

- MANO left/right: 778 vertices, 1,538 faces, 16 joints.
- SMPL-H: 6,890 vertices and 52 joints.
- SMPL-X: 10,475 vertices and 55 joints.
- SMPL-X accepts a 90D hand pose.

Changing one hand-pose channel produces a finite hand-vertex change:

```text
max vertex delta:  0.006592
mean vertex delta: 0.000022
```

This confirms that the installed asset stack can generate articulated hand
vertices once valid hand parameters are supplied.

Gate result:

```text
E3B hand-model assets: PASS
```

## Decision

The project cannot begin finger-level WM training yet. The blocker is data,
not model capacity or the existing body-model assets. The next required
action is to obtain raw BEHAVE hand pose if still available, or switch to
GRAB/ARCTIC for articulated hand supervision and external validation.

Until the raw-data and processed-cache gates pass, do not create a finger
world model, run HandX teacher evaluation, or add finger residuals to the
MaMi action.

## Evidence

Audit implementation:

```text
scripts/audit_hand_assets.py
```

Server audit:

```text
/root/autodl-tmp/mamihoi/outputs/e3_data_gate_20260918/audit.json
```

