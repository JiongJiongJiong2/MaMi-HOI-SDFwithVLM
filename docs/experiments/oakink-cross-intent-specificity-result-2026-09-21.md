# OakInk Cross-Intent Specificity Result

Date: 2026-09-21

Status: complete with specificity gate PASS

Protocol:

```text
docs/experiments/oakink-cross-intent-specificity-protocol-2026-09-21.md
```

Summary:

```text
docs/experiments/oakink_cross_intent_specificity_summary_v1_20260921.json
```

## Negatives

Non-handover OakInk sequences:

```text
use:    193 train / 79 val / 68 test
hold:    60 train / 70 val / 59 test
liftup:  42 train / 60 val / 54 test
```

Cross-intent release windows:

```text
val:  315
test: 150
```

The handover logistic model and threshold were fixed on train only.

## Results

| Split | Handover Recall | Cross-Intent FPR | Combined AUPRC | Combined F1 |
|---|---:|---:|---:|---:|
| val | 0.8593 | 0.0222 | 0.7948 | 0.7945 |
| test | 0.8523 | 0.0000 | 0.9123 | 0.8711 |

Cross-intent score distributions are low:

```text
val median:  0.0040, q75 0.0139
test median: 0.0013, q75 0.0102
threshold:   0.6526
```

## Gate

```text
val handover recall >= 0.50:                  pass
val cross-intent FPR <= 0.10:                 pass (0.0222)
val combined AUPRC above prevalence:          pass
test handover recall >= 0.50:                 pass
test cross-intent FPR <= 0.10:                pass (0.0000)
test combined AUPRC above prevalence:         pass
overall:                                      PASS
```

## Interpretation

The distance-history model does not collapse into a generic
contact-release detector. On held-out cross-intent sequences it fires
rarely, while retaining high handover recall on val and test.

## Remaining Boundary

The specificity test uses event windows centered on known release
events. It still does not prove that the model can scan a long sequence
and autonomously decide both when to emit a candidate and when to
suppress ordinary motion frames.

The next required step is a full-sequence online detector with:

```text
sliding history windows;
event-level precision and recall;
matched non-handover sequence durations;
participant-disjoint val/test;
event matching tolerance.
```

## Artifacts

Server:

```text
/root/autodl-tmp/oakink_cross_intent_specificity_v1_20260921
```

Local review:

```text
C:\Users\何炯乐\Documents\HOI项目\oakink_cross_intent_specificity_v1_20260921\server_output
```

## Implementation Hashes

```text
scripts/evaluate_oakink_cross_intent_specificity.py
fcbd51dd6f1934472cfeac4aa5958ee430b4b44f3e241a176db4f736325fe5be

tests/test_evaluate_oakink_cross_intent_specificity.py
baf2f0435944d7dc170dd34b920f0909bb07eff27ecb08667fa31a4cc25b8af3

summary
ac50a534af75f15bea6331cb6af1da6ff12ca166cf0a83a58cc3b39a678c54d7
```
