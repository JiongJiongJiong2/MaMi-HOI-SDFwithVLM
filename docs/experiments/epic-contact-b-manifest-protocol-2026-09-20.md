# EPIC-Contact B Manifest Protocol

Date: 2026-09-20

Status: frozen before full extraction

## Goal

Create the compact event table needed for contact-memory research without
copying the 7.1 GB EPIC-Contact training pickle to the server.

## Inputs

```text
E:/HOI/TMP/hopformer_repro_pkls/training_pkls/
epic_contact_v3_01-06-26_parallel_train_20260611.pkl

E:/HOI/TMP/hopformer_repro_pkls/training_pkls/
epic_contact_v3_01-06-26_parallel_test_20260611.pkl
```

The official HOPformer contact threshold is `3 mm`.

## Streaming Constraint

The local machine has less than 5 GB free physical memory. The training
pickle is 7.1 GB and must not be loaded as one object.

The builder uses the standard-library pure-Python unpickler, intercepting
top-level dictionary items as they are appended. Each sample is reduced to
small scalar records before large arrays enter the memory memo. The resulting
manifest is written as gzip-compressed JSONL.

## Frame Manifest

One row per `(split, video_id, frame_num)` after merging the separate left
and right hand records:

```text
split, video_id, frame
left/right validity
left/right contact at 3 mm
left/right minimum hand-object distance
left/right clip id and object name
same-object bimanual flag
same-object both-contact flag
```

## Episode Manifest

One row per contact run within a hand-specific clip:

```text
split, video_id, clip_id, hand, object_name
onset frame, last contact frame
contact frame count and span
maximum frame gap
onset observed / release observed
mean minimum hand-object distance
```

A run touching the start or end of a sampled clip is not counted as an
observed onset or release.

## Splits

The official train and test pickle files are labeled separately. No test
row may be used for model selection.

## Outputs

```text
epic_contact_frames_v1.jsonl.gz
epic_contact_episodes_v1.jsonl.gz
summary.json
```
