# E5-T Update Direction Decomposition Protocol

Date: 2026-09-20

Status: frozen before execution

## Goal

Test whether the contact regression from E5 to E5-T is explained by
hand vertices moving away from the object along the local contact normal.

This is a CPU-only diagnostic. It does not retrain a model, optimize a new
trajectory, change the E5/E5-T thresholds, or read the test split.

## Frozen Inputs

```text
batch:
/root/autodl-tmp/contact_action_20260914/sequence_contactopt_v1_20260919/cases/batch_summary.json

geometry:
/root/autodl-tmp/contact_action_20260914/sequence_contactopt_v1_20260919/geometry

E5:
/root/autodl-tmp/contact_action_20260914/e5_solver_v1_20260919

E5-T:
/root/autodl-tmp/contact_action_20260914/e5t_solver_v1_20260919
```

The run keeps the existing 49 chunks, 376 train/dev windows, sequence
boundaries, and E5/E5-T contact gates unchanged. The 28 test windows remain
unread.

## Reconstruction

For each chunk:

1. load the frozen E5 and E5-T finger-pose arrays;
2. run the same right-hand MANO layer used by ContactOpt;
3. validate the reconstruction against the stored ContactOpt vertices on
   the first and last frame;
4. use the per-frame object vertices from the frozen geometry cache.

The reconstruction gate is a maximum coordinate error of `1e-6 m`.

## Direction Decomposition

For every hand vertex at frame `t`:

```text
q_t = nearest object vertex to E5 hand vertex p_t
d_t = ||p_t - q_t||
n_t = (p_t - q_t) / d_t
delta_t = p_E5T(t) - p_E5(t)
normal_t = dot(delta_t, n_t)
tangent_t = ||delta_t - normal_t * n_t||
```

Positive `normal_t` means that E5-T moves the hand vertex away from the
nearest object vertex. Only vertices with `d_t <= 0.02 m` contribute to the
primary near-contact statistics.

The diagnostic reports:

- signed normal displacement;
- positive outward and negative inward components;
- tangential displacement;
- total displacement;
- outward and inward vertex fractions;
- mean nearest-object distance change.

This nearest-vertex normal is a geometric proxy consistent with the
existing ContactOpt distance cache. It is not a signed-distance field,
friction model, or force measurement.

## Frozen Groups

The contact gate is taken from the frozen E5 and E5-T result JSON:

```text
contact stable:      E5 pass and E5-T pass
contact regression:  E5 pass and E5-T fail
contact gain:        E5 fail and E5-T pass
contact common fail: E5 fail and E5-T fail
```

The primary comparison is `contact regression` minus `contact stable`.

## Statistics

All frame values are first averaged within an eight-frame window. Group
means use windows, and confidence intervals use 5,000 sequence-clustered
bootstrap resamples. Overlapping windows from one sequence remain one
cluster.

## Decision

The direction hypothesis receives GO only if the lower bound of the 95%
interval for:

```text
contact regression minus contact stable
near_signed_normal_mm
```

is greater than zero.

If GO, the next experiment is a direction-constrained smoothing/contact
projection pilot with matched temporal and pose budgets. If NO-GO, the
simple outward-normal explanation is rejected; the next diagnostic should
examine contact-correspondence changes or optimization drift rather than
building a projection layer on this premise.

## Outputs

```text
frame cache:
<output-dir>/frames/<chunk_id>.json

summary:
<output-dir>/summary.json
```

The script may be resumed after an SSH interruption without recomputing
completed chunks.
