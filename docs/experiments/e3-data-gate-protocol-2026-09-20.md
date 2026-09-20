# E3 Local Data Gate Protocol

Date: 2026-09-20

Status: frozen before execution

## Goal

Decide whether the locally available data can support:

- B: contact-memory lifecycle, including stable hold, transition, and
  release;
- C: future handover or release requirements affecting the current grasp.

This is a read-only CPU audit. It does not download gated data, train a
model, or access the held-out test split.

## Sources

```text
MaMi processed data:
/root/autodl-tmp/mamihoi/data/processed_data

HandX mixed hand-motion data:
/root/autodl-tmp/external/handx_data/MixData_no_arctic_h2o

HOPformer / EPIC-Contact code and access metadata:
/root/autodl-tmp/external/HOPformer
```

The audit distinguishes source availability from usable supervision:

- an official repository or processing script is not event data;
- hand-hand interaction text is not hand-object contact supervision;
- predicted ContactOpt geometry is not ground-truth object response;
- a gated dataset is blocked until its terms are accepted and files are
  locally inspectable.

## Required Fields

For B, a source needs complete temporal sequences with:

```text
hand pose + object state or mesh + contact transition/release evidence
```

For C, a source additionally needs:

```text
bimanual state + comparable initial/final object pose + role change,
handover, or release target
```

## Outputs

For each source, record:

- local availability and sequence counts;
- sample keys and tensor shapes;
- whether articulated hand pose is present;
- whether object state, mesh, contact, release, and handover are present;
- repository commit, license family, and access status;
- B and C gate decisions.

## Decision Rule

```text
B ready: complete event data is local and exposes contact transitions
C ready: B prerequisites plus bimanual handover or release evidence
otherwise: acquire and audit event data before modeling
```

No engineering work on contact memory or handover starts while both gates
are blocked.
