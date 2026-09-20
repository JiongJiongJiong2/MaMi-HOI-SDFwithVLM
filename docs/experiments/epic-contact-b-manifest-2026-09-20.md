# EPIC-Contact B Manifest Result

Date: 2026-09-20

Status: complete on the 60 GB server instance

Protocol:

```text
docs/experiments/epic-contact-b-manifest-protocol-2026-09-20.md
```

## Outputs

```text
docs/experiments/epic_contact_frames_v1.jsonl.gz
docs/experiments/epic_contact_episodes_v1.jsonl.gz
docs/experiments/epic_contact_b_manifest_summary_v1_20260920.json
```

Server outputs:

```text
/root/autodl-tmp/epic_contact_b_manifest_v1_20260920
```

## Inputs

The uploaded files matched the local SHA-256 identities:

```text
train:
121477112249faa7669a689bf22aa19ee60292ccce6059d6d6565cbc5bc90f5d

test:
35df2fea120b3676dc13d6b2d5a7c215c4e11b06ed35e102de7cda08d74456a4

keys:
e30e8fb92202da0251905e77ddaf24f89a2dae2a16dc1a9edacd9f373c6664b1
```

The build ran on the new server instance with:

```text
18 CPU cores
60 GB RAM
RTX 4090 D
standard C pickle loading
```

The complete train plus test build finished in roughly two minutes.

## Manifest Counts

```text
train records:                 55,983
test records:                   6,316
merged frames:                 57,686
contact episodes:              37,162
same-object bimanual frames:    3,975
both-contact frames:            3,907
```

Episode boundary observation counts:

```text
train onset=false release=false: 33,237
train onset=false release=true:      16
train onset=true  release=false:     20
train onset=true  release=true:      17
test  onset=false release=false:  3,869
test  onset=false release=true:        2
test  onset=true  release=false:       1
```

Most episodes touch a clip boundary, so their onset or release is
truncated. Only runs with both flags true are complete bounded
onset-hold-release examples.

## Output Hashes

```text
frames:
aa1fec2d7346c44e48894cc4c1320797a115a4b4d120beaf2f1859e78aaed875

episodes:
2007dec5bc7f84f9503cc74312fbc4cc7d39a4970cc2537b0c3a17d5f3fd104d
```

Both compressed files were downloaded to the repository and validated
line by line:

```text
frames rows:    57,686
episodes rows:  37,162
```

## Prepared Decisions

```text
B manifest:                    GO
complete onset-release pairs:  17
truncated boundary episodes:  37,128
C handover benchmark:          still NO-GO
```

The manifest is ready for contact-lifecycle analysis and model code. The
complete onset-release pair count is `17` in train and `0` in test. This
small number means release-specific claims should use a separately defined
transition dataset or a different data source. The 3,907
simultaneous-contact frames remain bimanual support evidence, not handover
evidence.

## Implementation Hash

```text
scripts/build_epic_contact_b_manifest.py
803c4b81e3fa0f08a0fc0c4d1698d7e89a0c790b3431dab11ae80db2d4bcfba5
```
