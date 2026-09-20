# Server Storage Audit

Date: 2026-09-20

Scope: read-only audit of `/root/autodl-tmp`.

## Capacity

```text
/root/autodl-tmp: 50 GB total, 40 GB used, 11 GB available (79%)
```

No files were deleted or moved during this audit.

## Largest Directories

| Path | Size | Current role | Recommendation |
|---|---:|---|---|
| `/root/autodl-tmp/external` | 19 GB | ContactOpt, HandX, models, caches | split by component |
| `/root/autodl-tmp/mamihoi` | 17 GB | processed data, outputs, checkpoint | keep main MaMi baseline |
| `/root/autodl-tmp/contact_action_20260914` | 3.8 GB | E1-E5 candidate and analysis artifacts | keep compact seed-1 evidence |

Within `external`:

| Path | Size | Recommendation |
|---|---:|---|
| `ContactOpt/data` | 10 GB | do not delete until E5 reproducibility is reclassified |
| `handx_data/MixData_no_arctic_h2o` | 6.0 GB | removable for the EPIC-Contact B line |
| `contactopt-venv2` | 1.2 GB | keep while ContactOpt may be rerun |
| `HandX_models` | 1.0 GB | optional archive; not needed for EPIC B |

Within `mamihoi`:

| Path | Size | Recommendation |
|---|---:|---|
| `data/processed_data` | 13 GB | keep; current MaMi data and SDF caches |
| `outputs/sdf_gate0` | 3.0 GB | prune debug weights first |
| `outputs/hoidyn_mami` | 648 MB | archive after checking report needs |
| `outputs/dsr01_repair_20260913` | 482 MB | archive after checking report needs |
| `checkpoints/model-9.pt` | 213 MB | keep |

## Immediate Reclaim Candidate

The safest cleanup is the set of old single-step debug weights under:

```text
/root/autodl-tmp/mamihoi/outputs/sdf_gate0
```

Twelve directories each contain about 214 MB of weights:

```text
U0_anomaly_baseline_1step_20260911
U0_component_debug2_1step_20260911
U0_finetune_debug_1step_20260911
U0_fp32_debug_1step_20260911
U0_grad_diag_1step_20260911
U0_overflow_debug_1step_20260911
U1U_fp32_candidate_debug_1step_20260911
U1U_fp32_chunk_debug_1step_20260911
U1U_fp32_debug_1step_20260911
U1U_fp32_timing_10step_20260911
U1U_geometry_debug_1step_20260911
U1U_smoke_300_fp32_candidate_20260911
```

Estimated reclaim: approximately `2.6 GB`.

These runs are debugging history. The metrics and report JSON should be
preserved, but the intermediate weights are not the frozen baseline.

## Second Reclaim Candidate

`MixData_no_arctic_h2o` is 6.0 GB of bimanual hand-only motion. It has no
object state, contact, release, or handover labels and is not needed for
the EPIC-Contact B manifest.

Estimated reclaim: `6.0 GB`.

The local workspace already contains the 5.07 GB train component. If the
server copy is removed, the 620 MB test component should be copied or
recreated first if HandX synthesis returns as a future branch.

## Do Not Delete Yet

`external/ContactOpt/data` contains about 10 GB of
`optimized_sequence_dataset_v1_*.pkl` files. They are intermediate
ContactOpt runs, but also the exact object/hand inputs needed to rerun the
E5 path. They are not needed for EPIC-Contact B, but should be archived
externally or deleted only after E5 is formally frozen as no longer
reproducible from raw steps.

The three large SMPL-X gender models in `processed_data/smpl_all_models`
must remain until all MaMi and ARCTIC preprocessing paths are confirmed to
use only one gender.

## Recommendation

```text
delete now after approval:  debug weights                 ~2.6 GB
optional second step:       HandX MixData                 ~6.0 GB
archive first:              ContactOpt optimized pickles   ~10 GB
```

The EPIC-Contact B manifest is small and does not require freeing server
space before construction. Cleanup can proceed independently, but no
deletion has been performed in this audit.
