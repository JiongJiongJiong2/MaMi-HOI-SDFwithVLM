# ContactOpt Cross-Object Replication Protocol: Phase 11

Date: 2026-09-18

Status: completed

Result report:

```text
docs/experiments/contactopt-phase11-replication-2026-09-18.md
docs/experiments/phase11_cohort/cohort_summary.json
docs/experiments/phase11_cohort/*.json
```

## Goal

Test whether the corrected ContactOpt teacher result replicates on a
pre-registered set of new sequences. Phase 10 selected one lexicographically
first candidate per object. Phase 11 excludes every sequence used by the
Phase 7, Phase 8, and Phase 10 calibrations, then again selects one
lexicographically first candidate per object.

This tests additional candidates from the two available subjects and adds
`clothesstand`, `largebox`, and `trashcan`, which were absent from the Phase
10 saved-vertex cohort.

## Frozen Manifest

Manifest:

```text
docs/experiments/contactopt_phase11_manifest.json
```

SHA-256:

```text
8c679a5e80bca1f63842819e988ba28ed94402ab3cbc425352577ead04d65fa5
```

Selection command:

```text
python build_contactopt_candidate_manifest.py \
  --root /root/autodl-tmp/contact_action_20260914 \
  --object monitor largetable plasticbox smallbox floorlamp whitechair \
           clothesstand largebox trashcan \
  --exclude-sequence sub17_monitor_007 sub17_monitor_026 \
                     sub16_largetable_000 sub16_largetable_015 \
                     sub16_plasticbox_001 sub17_smallbox_005 \
                     sub17_floorlamp_004 sub16_whitechair_012 \
  --output-json phase11_manifest.json
```

Selected sequences:

```text
sub17_monitor_015
sub16_largetable_001
sub16_plasticbox_004
sub17_smallbox_015
sub17_floorlamp_007
sub16_whitechair_002
sub16_clothesstand_001
sub16_largebox_005
sub16_trashcan_005
```

No ContactOpt outcome was used to choose paths, frames, objects, or
thresholds.

## Frozen Method

Use the same protocol as Phase 10:

```text
p10_distance <= 0.02 m
at least 3 eligible frames per case
required successful frames = max(3, ceil(0.7 * eligible frames))
w_opt_rot = 0
w_opt_trans = 0
w_obj_rot = 0
rand_re = 0
250 iterations
```

Undefined ContactOpt capsule values are treated as zero contact. Finite
fractions are recorded for every input and refined hand array.

## Promotion Gate

Promotion requires all of:

1. at least 7 of 9 requested objects have a saved-vertex candidate;
2. at least 6 cases have at least 3 eligible frames;
3. at least 5 eligible cases improve contact in at least 70% of eligible
   frames;
4. every invoked case has wrist drift at most `0.01 mm` and object drift at
   most `0.001 mm`;
5. no invoked case has mean nearest-distance regression greater than 10%.

Decision rule:

```text
5 or more eligible contact successes: GO for teacher integration study
3 to 4 eligible contact successes: CONDITIONAL, inspect per-frame failures
2 or fewer eligible contact successes: NO-GO for broad teacher integration
```

Failure of the orientation-specific contact metric while nearest distance
still improves is preserved as a failure mode; it will not be relabeled as
success.

## Boundaries

This remains a kinematic refinement teacher test. It does not create object
motion, recover MaMi `pose_hand`, or justify finger world-model training.
