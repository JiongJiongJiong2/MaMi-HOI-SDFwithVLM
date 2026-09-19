# Sequence-Level Contact Dataset Result

Date: 2026-09-19

Status: complete on the AutoDL no-card server

Protocol:

```text
docs/experiments/contactopt-sequence-dataset-protocol-2026-09-19.md
```

Raw manifest:

```text
docs/experiments/contactopt_sequence_dataset_v1_20260919.json
```

Manifest SHA-256:

```text
0605026647b1dabb44ee13723d16c97297524f7ffc2df3083665512eaf541054
```

## Server Result

The CPU-only scan found 70 deterministic saved-vertex candidates after
sequence deduplication and produced 404 eligible eight-frame windows.

| Split | Assigned sequences | Sequences with windows | Objects with windows | Windows | Frames |
|---|---:|---:|---|---:|---:|
| train | 46 | 34 | 8 | 256 | 2,048 |
| dev | 17 | 15 | 9 | 120 | 960 |
| test | 7 | 4 | 2 | 28 | 224 |

All previously inspected Phase 7, 8, 10, 11, 12, and 15 sequences are
assigned to `dev`. The test split has no overlap with those calibration or
development sequences.

The test split contains only `monitor` and `largetable` sequences. That is a
real coverage limitation: the dataset can support temporal-solver
development, but it cannot by itself establish cross-object generalization.

## Decision

Keep this manifest as `sequence_dataset_v1` and use it to implement the
sequence-level contact and smoothness objective. Do not spend the test split
until the objective, candidate execution, and checkpoint rule are frozen.

The next GPU stage should generate ContactOpt outputs only for train/dev
windows. Test outputs must remain ungenerated and unread until the final
confirmation run.

This result does not train a world model and does not establish that
ContactOpt output is temporally natural.
