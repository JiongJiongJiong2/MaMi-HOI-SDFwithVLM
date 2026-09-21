# C0-R1 NO-GO Audit

Date: 2026-09-21

Status: implementation correct; frozen NO-GO valid; scientific scope
must be narrowed

Source result:

```text
docs/experiments/oakink-c0r-oracle-reranking-result-2026-09-21.md
```

## Question

Determine whether the C0-R1 NO-GO reflects a real failure, an
implementation error, or a protocol/measurement problem.

## Implementation Audit

The implementation follows the frozen protocol:

```text
candidate object-frame alignment is correct;
no-future uses contact score;
correct and shuffled conditions use the same candidate set;
top-1 selection and grouped bootstrap follow the frozen formulas;
the gate is applied without a numerical error.
```

No code bug was found that would reverse or invalidate the reported
NO-GO.

## Independent Server Audit

Paired top-1 transitions:

| Comparison | Gains | Losses | Exact McNemar p |
|---|---:|---:|---:|
| correct vs no-future | 7 | 1 | 0.0703 |
| correct vs shuffled | 12 | 0 | 0.00049 |

Among the eight decisions changed from no-future, seven select the
observed target-event giver and one does not.

The evaluation delta is:

```text
positive: 9
zero:     42
negative: 3
mean:     0.05806
```

The grouped bootstrap interval is positive, but the effect is sparse.

## Candidate Availability

| Eligibility | Events | Objects with >=2 | Eligible Targets | Candidate Counts |
|---|---:|---:|---:|---|
| complete window >=45 | 82 | 27 | 54 | all 2 |
| reference frames valid | 106 | 41 | 82 | all 2 |
| all C0 events | 106 | 41 | 82 | all 2 |

The complete-window rule halves the number of eligible targets relative
to a reference-frame-valid rule. It does not reduce candidate count per
object because every usable object group contains exactly two events.

This is an underpowering issue, not evidence that a richer candidate
set exists in the current C0 arrays.

## Evaluation Overlap

The condition and evaluation reuse the same target receiver contact
region and receiver hand:

```text
condition:
  target region clearance;
  target receiver hand clearance.

evaluation:
  target region clearance;
  target receiver hand clearance;
  candidate contact coverage.
```

Independent recomputation gives:

```text
same per-event argmax:                 45 / 54 = 0.8333
mean within-event feature correlation: 0.9429
```

The strong correct-versus-shuffled result is therefore not fully
independent evidence. It mostly confirms that the correct target is
more compatible with its own receiver geometry than a donor target.

## Gate-Design Audit

Two frozen criteria are difficult to interpret as direct utility tests:

1. `selection change >= 25%` measures how often the condition changes
   the decision, not whether changed decisions are beneficial.
2. `secondary direction consistency` requires even the `0.25` weight to
   change at least 25% of selections. A low weight is expected to
   produce few changes and is not a meaningful robustness requirement
   for this objective.

The observed changed decisions are actually favorable:

```text
7 of 8 changed selections move toward the observed target-event giver.
```

This does not rescue the frozen result because the paired top-1 gain
against no-future is not statistically significant (`p=0.0703`) and the
evaluation is not independent.

## Verdict

```text
C0-R1 implementation:             valid
frozen promotion decision:         NO-GO
global future-conditioning claim:  not disproven
current protocol as evidence:      underpowered and partly circular
full optimizer authorization:      not granted
```

The correct description is:

> The current observed-grasp reranking protocol did not meet its frozen
> promotion gate. This does not establish that future-conditioned grasp
> selection is ineffective.

## What Would Be Required for C0-R2

A corrected experiment would need a new frozen protocol with:

1. all reference-valid events, giving 82 targets instead of 54;
2. condition features using receiver-target geometry only;
3. evaluation using held-out receiver geometry such as wrist/palm or
   joints that do not feed the condition;
4. direct paired utility tests instead of a selection-change-frequency
   gate;
5. explicit acceptance of the pairwise candidate limitation, or a new
   source of multiple candidate grasps.

This is a new experiment, not a repair of the frozen C0-R1 result.

## Impact on Other Branches

| Branch | Impact |
|---|---|
| C oracle optimization | Full optimizer remains unauthorized under current evidence; a corrected C0-R2 is possible |
| D action-conditioned WM | Not blocked by C0-R1; it is independently blocked by the lack of same-initial-state multi-action outcomes |
| B contact memory/lifecycle | Not blocked; remains an independent mechanism question |
| E5/E5-T sequence refinement | Not blocked; remains an engineering baseline |
| OakInk handover detection | Not invalidated; detection was only used as event extraction infrastructure |

## Server Evidence

```text
/root/autodl-tmp/oakink_c0r_result_audit_v1_20260921/audit.json
```

Three audit unit tests passed on the no-card server.

## Implementation Hashes

```text
scripts/audit_oakink_c0r_result.py
d3358a44501135e8abd3c5b688fe72b50c09afc70a0b0cef2d1149ae15cbc891

server audit.json
1c99fe45b0ab26358899cf817b8dd90aed76e1021992b5d8d998b320b582130d
```
