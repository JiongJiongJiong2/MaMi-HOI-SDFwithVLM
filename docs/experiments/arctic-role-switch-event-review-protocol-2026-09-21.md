# ARCTIC Role-Switch Event Review Protocol

Date: 2026-09-21

Status: frozen before event-feature extraction

## Goal

Describe whether Tier A role-switch candidates occur during observable
object motion and hand relocation rather than as static contact
alternation.

This review does not turn candidates into handover labels.

## Inputs

```text
/root/autodl-tmp/arctic_role_switch_quality_v1_20260921/
  tier_a_candidates.jsonl.gz

/root/autodl-tmp/arctic_data/data/raw_seqs/
```

Primary threshold is `3 mm`.

## Event Window

For each Tier A candidate:

```text
pre window:  15 frames before outgoing release
post window: 30 frames after receiving onset
```

## Features

The review records:

```text
object translation change
object rotation change
outgoing-hand root translation before release
outgoing-hand root retreat after release
receiving-hand root displacement after onset
left-right hand-root separation change
```

Object translation and hand-root motion are measured in meters. Object
rotation is measured in degrees.

## Descriptive Thresholds

The analysis reports counts, quantiles, and the fraction of candidates
with:

```text
object translation change >= 5 mm
object rotation change >= 5 degrees
either object-motion condition
outgoing-hand retreat >= 5 mm
receiving-hand displacement >= 5 mm
```

These thresholds are descriptive and are not a new pass/fail gate.
