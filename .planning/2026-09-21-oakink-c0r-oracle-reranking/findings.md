# Findings and Decisions

## Inputs

Completed C0 dataset:

```text
/root/autodl-tmp/oakink_c0_event_dataset_v1_20260921
```

C0 established:

```text
106 events
82 complete events
41 same-object groups
21 same-object conditioning-contrast groups
79 hard-control events
```

## Constraints

- Run all new experiments on the server.
- Do not use the current OakInk test as a fresh final benchmark.
- Do not infer causality from observed successful handovers.
- Keep object-frame alignment explicit and deterministic.
- Keep the primary score and gate frozen before viewing the result.

## Design Decisions

| Decision | Rationale |
|---|---|
| Candidate set is observed complete events sharing the same object id | The data contains no same-initial-state multi-action outcomes |
| Candidate giver hands are expressed in the object canonical frame | Removes the target event's camera pose from the ranking |
| Correct and shuffled conditions use the same candidate set | Isolates whether receiver-target identity changes the ranking |
| Shuffled target is drawn deterministically from the same object | Prevents object-category mismatch from creating a trivial signal |
| Evaluation uses target receiver geometry beyond the condition region | Reduces direct score/evaluation circularity |

## Result

The frozen gate is NO-GO:

```text
eligible target events:      54
selection change rate:       0.1481
required selection change:   0.25
correct minus shuffled mean: 0.05806
95% grouped CI:              [0.00608, 0.12886]
overall:                     NO-GO
```

The correct future target has a measurable positive effect over the
shuffled target, but it changes the primary top-1 decision too rarely.
The observed candidate universe contains exactly two grasps per target.
