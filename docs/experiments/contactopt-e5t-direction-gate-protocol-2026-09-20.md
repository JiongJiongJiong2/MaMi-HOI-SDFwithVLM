# E5-T Direction Gate Protocol

Date: 2026-09-20

Status: revised before successful execution

## Goal

Test whether the E5-to-E5-T near-surface outward direction is useful as a
minimal discrete intervention before implementing a continuous
direction-constrained projection.

The selector may keep E5-T or fall back to E5. It cannot use E5-T contact
or temporal gate labels as inputs.

## Frozen Inputs

```text
E5 result:
/root/autodl-tmp/contact_action_20260914/e5_solver_v1_20260919/result.json

E5-T result:
/root/autodl-tmp/contact_action_20260914/e5t_solver_v1_20260919/result.json

direction summary:
/root/autodl-tmp/contact_action_20260914/e5t_direction_decomposition_v1_20260920/summary.json
```

The same 49 chunks and 376 train/dev windows are used. The 28 test windows
remain unread.

## Selector

The primary risk feature is:

```text
near_outward_fraction
```

For each window:

```text
if E5 contact gate did not pass:
    keep E5-T
else if the feature is missing or non-finite:
    keep E5-T
else if feature > threshold:
    select E5
else:
    keep E5-T
```

The threshold is fitted on train windows only.

The missing-feature fallback was changed from E5 to E5-T before the first
successful execution. The original rule selected more than the 20-window
budget in train, so no threshold was feasible. No train or dev outcome
was produced before this revision.

## Train-Only Threshold Rule

Candidate thresholds are the midpoints between adjacent finite train
feature values, plus positive infinity.

The fitted threshold must select at most:

```text
floor(0.08 * train_window_count) = 20
```

Among valid thresholds, maximize:

```text
combined_pass_count - 0.05 * selected_e5_count
```

Ties prefer the larger threshold. No dev or test value participates in
threshold selection.

## Outcomes

For train and dev, report:

- selected E5 count;
- contact pass count;
- combined pass count;
- always-E5 and always-E5-T baselines;
- oracle combined count;
- paired selected-minus-E5-T combined count with a 5,000-draw
  sequence-clustered bootstrap interval.

Leave-one-object-out threshold selection is reported as a stability check
inside train.

## Gate

The gate passes only if all of the following hold:

1. selected combined passes exceed always-E5-T on train;
2. selected combined passes exceed always-E5-T on dev;
3. selected contact passes are not below always-E5-T on dev;
4. the dev lower bootstrap bound for selected-minus-E5-T combined passes
   is greater than zero.

Failure of condition 4 is a statistical-power or robustness failure, not
permission to tune on dev.

## Validation Boundary

The dev split was inspected during the preceding direction pilot. This run
is therefore a locked-threshold confirmation on dev, not a pristine final
test. Final test evaluation remains prohibited until the selector or
projection design is frozen after this result.
