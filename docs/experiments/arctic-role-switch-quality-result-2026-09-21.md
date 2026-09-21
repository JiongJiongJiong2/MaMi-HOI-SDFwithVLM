# ARCTIC Role-Switch Quality Result

Date: 2026-09-21

Status: pilot Tier A gate PASS; official ARCTIC test unavailable

Protocol:

```text
docs/experiments/arctic-role-switch-candidate-quality-protocol-2026-09-21.md
```

Upstream data gate:

```text
docs/experiments/arctic-handover-data-gate-protocol-2026-09-21.md
```

## Raw Data Gate

The server job completed with:

```text
exit code:              0
sequences processed:  301
failures:               0
pilot raw gate:       PASS
```

At the official ARCTIC `3 mm` contact threshold:

| Split | Raw Candidates | Candidate Sequences | Participants |
|---|---:|---:|---:|
| train | 1,182 | 222 | 8 |
| val | 143 | 32 | 1 |

The raw rule intentionally uses a wide `-15/+15` onset window and
therefore includes ordinary bimanual support and temporary switching.

## Tier Definitions

All tiers require at least 15 stable receiving-contact frames after the
outgoing release.

```text
Tier A: onset offset <= 5 frames, overlap <= 10 frames, outgoing gap >= 15
Tier B: onset offset <= 10 frames, overlap <= 20 frames, outgoing gap >= 15
```

Tier A is the primary pilot manifest.

## Primary 3 mm Results

### Tier A

| Split | Candidates | Sequences | Participants | Objects | Left-to-Right | Right-to-Left |
|---|---:|---:|---:|---:|---:|---:|
| train | 45 | 35 | 8 | 10 | 25 | 20 |
| val | 9 | 5 | 1 | 4 | 5 | 4 |

### Tier B

| Split | Candidates | Sequences | Participants | Objects | Left-to-Right | Right-to-Left |
|---|---:|---:|---:|---:|---:|---:|
| train | 119 | 85 | 8 | 11 | 57 | 62 |
| val | 15 | 10 | 1 | 8 | 10 | 5 |

The Tier A reduction from `1,182 -> 45` train candidates removes
transitions that overlap too long, start too far from release, or are
followed by immediate re-contact of the outgoing hand.

## Threshold Sensitivity

Tier A at each contact threshold:

| Threshold | Train Candidates | Train Participants | Train L-to-R / R-to-L | Val Candidates | Val L-to-R / R-to-L |
|---:|---:|---:|---:|---:|---:|
| 1 mm | 44 | 8 | 18 / 26 | 10 | 2 / 8 |
| 3 mm | 45 | 8 | 25 / 20 | 9 | 5 / 4 |
| 5 mm | 38 | 8 | 20 / 18 | 8 | 4 / 4 |
| 10 mm | 24 | 7 | 10 / 14 | 3 | 1 / 2 |

The candidate count is stable between `1-5 mm` and decreases at
`10 mm`. This supports a threshold-qualified pilot, but the exact
threshold remains part of the result.

## Gate

```text
train Tier A candidates >= 10:             pass
train Tier A participants >= 3:            pass
train Tier A both directions >= 2:         pass
train Tier A object categories >= 3:       pass
val Tier A candidates >= 1:                pass
pilot Tier A overall:                      PASS
official test available:                   false
final benchmark ready:                     false
```

Official test subject `s03` is withheld from the public raw release.
The result can support pilot event modeling, but cannot support an
official ARCTIC test claim.

## Artifacts

Server:

```text
/root/autodl-tmp/arctic_role_switch_quality_v1_20260921
```

Local review copy:

```text
C:\Users\何炯乐\Documents\HOI项目\arctic_role_switch_quality_v1_20260921
```

Files:

```text
candidate_quality_summary.json
tier_a_candidates.jsonl.gz
tier_b_candidates.jsonl.gz
```

## Decision

ARCTIC is viable for a pilot role-switch event manifest. The next step
is event-level validation: confirm that Tier A candidates represent
sustained object-contact role transfer rather than support-hand
switching, and derive participant-disjoint train/validation folds from
the accessible eight train subjects.

No handover model or final benchmark claim is authorized until that
event-level validation is complete.

## Implementation Hashes

```text
scripts/audit_arctic_handover_gate.py
aa5922cff40ac130c9cfd7407d34909118d6390edbe17512a7c22fd2993574c0

scripts/analyze_arctic_role_switch_candidates.py
b6e29f707680320e72e9cd5cf35f58fff7ceb2d8f25d5aaf793b4013d63c244f

tests/test_audit_arctic_handover_gate.py
d06287eae9a25033a1583abebf4a0fee894d29c38b9f079cb80bd484de8bb103

tests/test_analyze_arctic_role_switch_candidates.py
97d82a18c2126b31ef438241bd104c5a951df9b1efef8357840bc3ec53bfc4e4
```
