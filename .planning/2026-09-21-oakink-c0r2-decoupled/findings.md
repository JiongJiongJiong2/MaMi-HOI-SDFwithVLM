# Findings and Decisions

## Audit Inputs

C0-R1 audit established:

```text
condition and evaluation reused target region and receiver hand;
same argmax on 45 / 54 events;
mean within-event correlation 0.9429;
complete-window eligibility used 54 targets;
reference-valid eligibility provides 82 targets;
all candidate sets contain exactly two events.
```

## Repairs

| C0-R1 weakness | C0-R2 response |
|---|---|
| Circular evaluation | Condition uses object-frame receiver contact region only; evaluation uses receiver joints only |
| Unnecessary target loss | Use all reference-frame-valid events |
| Selection-change gate | Use paired evaluation utility and collision deltas |
| Low-weight consistency artifact | Require directional consistency only at the primary and stronger weights |

## Remaining Limitation

No object group has more than two observed candidates, so C0-R2 remains
a pairwise oracle test.

## Result

The decoupled C0-R2 gate is NO-GO:

```text
eligible events:             82
independent oracle top-1:    0.5732
no-future top-1:             0.5000
correct future top-1:        0.5000
shuffled future top-1:       0.5000
correct minus no-future:     0.0, 95% CI [0.0, 0.0]
correct minus shuffled:      0.0, 95% CI [0.0, 0.0]
overall:                     NO-GO
```

The independent oracle shows small headroom, but the receiver-region-only
condition does not change the primary ranking.
