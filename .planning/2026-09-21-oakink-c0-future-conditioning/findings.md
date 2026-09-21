# Findings and Decisions

## Requirements

- Prove that C0 can run without GPU before starting expensive work.
- Preserve the audited distinction between handover detection and
  downstream conditioning.
- Do not use the current OakInk test as a fresh final benchmark.
- Do not treat cross-person OakInk geometry as a MaMi action schema until
  the mapping is frozen.

## Server Audit

Connected container:

```text
autodl-container-77c34795d5-29846918
2026-09-21 20:03 CST
no-card mode, no GPU
/root/autodl-tmp: 45/50 GB used, 5.1 GB available
```

The container id matches the instance used earlier in the day.

Key server script hashes match the main repository:

```text
build_oakink_handover_manifest.py
50494eb89384a15a58df526455ba60b9f62d7f2b827be7f1019aa3a610e569ce

build_oakink_handover_windows.py
ad3d38ce033a5ad71e798ae7c9a51940890fab0b0149f225d13ff7fd6708fb79

evaluate_oakink_online_handover_detector.py
a2652e217f70b82e0b79bb92e78dc9ed54bc47fd9961d1c050506f971e4fd5c9

evaluate_oakink_cross_intent_specificity.py
fcbd51dd6f1934472cfeac4aa5958ee430b4b44f3e241a176db4f736325fe5be

evaluate_oakink_sequence_aware_handover_detector.py
f9a8f23c71f1cf147363bbc471a7cc6dcd75b11f2d1cf74ae7fa5e8c7a3ae378
```

Derived manifest hashes match the local audited copies:

```text
summary.json
53b6326aa57e0d1797276bbf49e73a59d07b7e55c1101975c7c1a299cc50574e

role_switch_candidates.jsonl.gz
65d012f221e9ebd34973dbf5136439e23cb829c88b36b03f095244eb46d00a2a

handover_trajectories.npz
4d2dba50a273975d9c83e0f5f2aca301a7610551c14af71bdfbae5b758895fe14
```

Structural verification:

```text
retained sequences: 126
split counts:       44 train, 41 val, 41 test
5 mm candidates:    110
candidate sequences: 33 train, 36 val, 37 test
failures:           0
all data gates:     pass
```

The C0 compact dataset was extracted from the local raw annotation archive
and uploaded before running the audit on the server:

```text
/root/autodl-tmp/oakink_c0_scripts
/root/autodl-tmp/oakink_c0_event_dataset_v1_20260921
```

Dataset hashes:

```text
events.npz
fb628f7c9e50d380ca11bf0a35c4c2d14a35ddd2fcb65a8b4ffe41210b5964f8

objects.npz
2ac3408005dc8cf818756c458ecc512c59f3ac14c7f8e0f0492e482403136304

targets.npz
64ec9fbcc267d9885f239f9d132f2ca7cb47e6ce1e94b5cfd8fc61beeec0c3f4

manifest.jsonl
dd823deef245e2244224ab1a4b5a7115a91364a71918fc95463f1245586e17d3
```

## Research Findings

OakInk gives explicit handover intent, giver/receiver identity, and
contact geometry. This is enough for event-centered data extraction.
It does not yet prove that conditioning an optimizer on the receiver
region changes the pre-transfer giver grasp.

## C0 Feasibility Result

The completed server audit passed all frozen data gates:

```text
primary events:                  106
complete events:                 82
complete split counts:           23 train, 30 val, 29 test
giver and receiver regions:      106 / 106
same-object groups:              41
same-object contrast groups:     21
hard-control events:             79
geometry maximum error:          1.64e-8 m
overall C0 data gate:            PASS
```

This authorizes the next oracle-optimization stage. It does not
establish that a future receiver condition improves the earlier grasp.

## Technical Decisions

| Decision | Rationale |
|---|---|
| Extract event arrays locally from raw zips | The raw package is 3.6 GB and should not consume the nearly full server data disk |
| Upload only the compact derived arrays and audit scripts | The server already has the matching event manifest |
| Separate C0 feasibility from oracle optimization | The feasibility result determines whether geometry-based optimization is justified |
| Run every post-extraction experiment on the server | The user requires experiment execution to remain on the AutoDL instance |
