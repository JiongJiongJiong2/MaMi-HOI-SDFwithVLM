# EPIC-Contact Data Gate Protocol

Date: 2026-09-20

Status: frozen before execution

## Goal

Determine whether the downloaded EPIC-Contact test payload can support:

- B contact-memory lifecycle;
- C future-handover research.

The audit uses the official EPIC-Contact contact threshold of `3 mm`.

## Inputs

```text
test payload:
E:/HOI/TMP/hopformer_repro_pkls/training_pkls/
epic_contact_v3_01-06-26_parallel_test_20260611.pkl

keys:
E:/HOI/TMP/hopformer_repro_pkls/training_pkls/
epic_contact_v3_01-06-26_parallel_20260611_keys.pkl
```

The 7.1 GB training payload is not materialized in memory for this gate.
The keys file provides the full sample count, while the test payload
provides schema and event statistics.

## Contact Definition

For a valid hand:

```text
contact = min(dist.<hand>o) <= 0.003 m
```

Invalid hands are non-contact. The audit reports both all-frame and
valid-hand fractions.

## Required Evidence

For B:

```text
contact starts and ends distributed across videos
```

For C:

```text
left-to-right or right-to-left contact-role change candidates
```

A role-change candidate is not an explicit handover label. It is only a
signal for a later annotation audit.

## Decision

```text
B candidate:
  contact schema present and both starts and ends observed

C candidate:
  role-change candidates observed, still requiring manual/schema review
```

No GPU is used and the training payload is not modified.
