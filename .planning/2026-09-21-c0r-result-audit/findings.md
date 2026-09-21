# Findings and Decisions

## Audit Questions

1. Are the reported C0-R1 numbers computed as specified?
2. Is the evaluation metric independent of the conditioning metric?
3. Did the complete-window eligibility rule reduce the candidate universe
   unnecessarily?
4. Does a failed C0-R1 promotion block D or unrelated B/E5 work?

## Initial Code Observations

- The implementation follows the frozen object-frame alignment.
- Candidate sets contain exactly two events in the reported run.
- The condition score uses the target receiver contact region and
  receiver hand.
- The evaluation score uses the same target receiver contact region and
  receiver hand, so the positive correct-versus-shuffled result is not
  fully independent evidence.
- The promotion gate requires a minimum selection-change frequency,
  which is a design choice rather than a direct measure of utility.

These observations require server-side quantification before the final
verdict.

## Server Audit Results

```text
correct vs no-future top-1:
  7 gains, 1 loss, exact McNemar p=0.0703

correct vs shuffled top-1:
  12 gains, 0 losses, exact McNemar p=0.00049

changed selections:
  8 total; 7 select the observed target-event giver

candidate universes:
  complete >=45: 54 targets, 27 object groups, all pairs
  reference-valid: 82 targets, 41 object groups, all pairs

condition/evaluation overlap:
  same argmax on 45/54 events
  mean within-event correlation 0.9429
```

## Verdict

The implementation is correct and the frozen gate is a valid NO-GO.
However, the result is underpowered and partly circular:

- the condition and evaluation reuse target receiver geometry;
- every candidate set contains exactly two observed grasps;
- the selection-change-frequency gate is not a direct utility test;
- the low-weight secondary consistency check is not informative.

The result rejects the current promotion claim, not future
conditioning as a research mechanism.
