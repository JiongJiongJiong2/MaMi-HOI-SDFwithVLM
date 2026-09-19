# Sequence-Level Contact Dataset Protocol

Date: 2026-09-19

Status: preregistered; server build in progress

## Goal

Replace the small set of manually inspected temporal windows with a
sequence-disjoint dense-window dataset that can support calibration and one
untouched validation of a sequence-level contact solver.

This step builds data only. It does not train a world model, does not generate
new ContactOpt outputs, and does not modify MaMi.

## Frozen Inputs

Scan the saved-vertex MaMi results under:

```text
/root/autodl-tmp/contact_action_20260914
```

Use the objects from the teacher replication:

```text
monitor largetable plasticbox smallbox floorlamp whitechair
clothesstand largebox trashcan
```

Keep the geometry-only eligibility rule:

```text
p10(hand-vertex-to-object-vertex nearest distance) <= 0.02 m
```

The following previously inspected sequences are assigned to `dev` and can
never enter `test`:

```text
sub17_monitor_007
sub17_monitor_026
sub16_largetable_000
sub16_largetable_015
sub16_plasticbox_001
sub17_smallbox_005
sub17_floorlamp_004
sub16_whitechair_012
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

## Frozen Selection Rule

For every candidate NPZ:

1. require `pred_right_hand_verts`, `pred_object_verts`, and `object_faces`;
2. deduplicate repeated sequences by the lexicographically first object/path;
3. compute per-frame p10 distance;
4. find contiguous runs satisfying the eligibility threshold;
5. require a run of at least eight frames;
6. enumerate eight-frame windows with stride four and include the final
   window of each run;
7. retain at most eight evenly spaced windows per sequence.

Assign all explicitly listed development sequences to `dev`. Split the
remaining sequences into `train` and `test` by a stable SHA-256 hash with
`test_fraction=0.2` and `split_seed=20260919`. Sequence membership must remain
disjoint across all three splits.

## Server Command

```bash
cd /root/autodl-tmp/mami-wm-c537e84
/root/autodl-tmp/external/contactopt-venv2/bin/python \
  scripts/build_contactopt_sequence_dataset.py \
  --root /root/autodl-tmp/contact_action_20260914 \
  --object monitor largetable plasticbox smallbox floorlamp whitechair \
           clothesstand largebox trashcan \
  --output-json /root/autodl-tmp/contact_action_20260914/contactopt_sequence_dataset_v1_20260919/manifest.json \
  --threshold-m 0.02 \
  --window 8 \
  --stride 4 \
  --min-run 8 \
  --max-windows-per-sequence 8 \
  --test-fraction 0.2 \
  --split-seed 20260919 \
  --dev-sequence sub17_monitor_007 \
                 sub17_monitor_026 \
                 sub16_largetable_000 \
                 sub16_largetable_015 \
                 sub16_plasticbox_001 \
                 sub17_smallbox_005 \
                 sub17_floorlamp_004 \
                 sub16_whitechair_012 \
                 sub17_monitor_015 \
                 sub16_largetable_001 \
                 sub16_plasticbox_004 \
                 sub17_smallbox_015 \
                 sub17_floorlamp_007 \
                 sub16_whitechair_002 \
                 sub16_clothesstand_001 \
                 sub16_largebox_005 \
                 sub16_trashcan_005
```

## Acceptance

The dataset is usable only when all of the following hold:

1. `valid_sequence_disjoint_dataset` is `true`;
2. train, dev, and test each contain at least one window;
3. no sequence appears in more than one split;
4. every window has exactly eight contiguous frames;
5. every selected frame satisfies the frozen p10 threshold;
6. train and test object coverage are reported before any solver is trained;
7. the JSON manifest is checksummed and the test split remains unread by
   model-selection code.

If any split has too few windows or too narrow object coverage, the split is
reported as inadequate rather than changing thresholds or mixing sequences.

## Next Step

Only after this dataset gate passes may a sequence-level objective be frozen.
The first objective must include both contact fidelity and temporal
smoothness, and must be selected on `train` plus `dev` only.
