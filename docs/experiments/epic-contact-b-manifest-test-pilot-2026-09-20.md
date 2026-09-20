# EPIC-Contact B Manifest Test Pilot

Date: 2026-09-20

Status: test manifest complete; full train manifest blocked by resources

Superseded by:

```text
docs/experiments/epic-contact-b-manifest-2026-09-20.md
```

Protocol:

```text
docs/experiments/epic-contact-b-manifest-protocol-2026-09-20.md
```

## Outputs

```text
docs/experiments/epic_contact_frames_test_v1.jsonl.gz
docs/experiments/epic_contact_episodes_test_v1.jsonl.gz
docs/experiments/epic_contact_b_manifest_test_summary_v1_20260920.json
```

The frame manifest has `6,165` merged frames from `6,316` left/right
records. The episode manifest has `3,872` contact runs at the official
`3 mm` threshold.

## Episode Observation Counts

```text
onset observed false, release observed false: 3,869
onset observed false, release observed true:      2
onset observed true,  release observed false:     1
```

Most runs touch a clip boundary and therefore have a truncated onset or
release. They remain valid contact-hold examples but must not be counted
as complete onset-release episodes.

## Hand and Object State

```text
same-object bimanual frames:       151
same-object both-contact frames:   151
```

These are simultaneous bimanual contacts, not sequential handover.

## Full Training Attempt

The training pickle is `7.11 GB`. The local machine has about 16 GB total
physical memory:

```text
test pickle:     0.81 GB
train pickle:    7.11 GB
keys:            0.003 GB
```

The streaming parser was validated on the full test payload. The training
attempt reached `25,000 / 55,983` records before memory growth and runtime
made continuation on the current machine unsafe. The run was stopped and
did not produce a partial training manifest.

The server instance has about 2 GB RAM and only 14 GB free storage. It
cannot load the training pickle, and the SSH gateway rejected SFTP and
reverse-tunnel transfer attempts. The compact manifest is therefore kept
in the repository while the full training manifest remains blocked.

## Decision

```text
test B manifest:              GO
training B manifest:          blocked by local memory and transfer limits
model training on train set:  not started
B model code readiness:       schema ready; data not ready
```

## Next Action

Choose one:

1. run the manifest builder on a machine with at least 32 GB RAM;
2. upload the training pickle to a server with at least 32 GB RAM and run
   the builder there;
3. split the training pickle with a converter on such a machine, then
   transfer only the three compact manifest outputs.

Do not start B model fitting from the test manifest alone.

## Implementation Hash

```text
scripts/build_epic_contact_b_manifest.py
d75dc04d4f24e6e96cced1bd085ec52e70bd6efab399099d614d4f7e4a6c4f58
```
