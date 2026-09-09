# U1 geometry supervision decision — 2026-09-08

2026-09-09 截面证据更新：**优先 A 不变，没有切换 B。** plasticbox 盒口部分分离双表面已追踪到原始三角面，不能把所有双线都归为重复面；trashcan 部分水平截面剥离悬挂线后可恢复单环，但尚无材料内外双环。“材料边界未建立”不等于“所有区域均无第二表面”。详见[中文截面定量报告](material-sections-audit-zh.md)。本次只更新证据，不解除 Gate，也不授权 source/cache 重建。

**Original signed U1: BLOCKED.** Pause regeneration of plasticbox/trashcan from
the currently available meshes. Continue an independent surface-query readiness
diagnostic; this does not implement or approve a replacement U1 objective.
No training, CUDA, mesh/SDF/cache replacement or evaluator change is authorized
by a successful CPU diagnostic. U2/U3/U4 and the novelty claims are unchanged.

## Decision and evidence

| Route | Minimum useful form | Current decision |
|---|---|---|
| A: trustworthy signed SDF | Independently defined material boundary, signed Euclidean distance, then a validated training cache | Best controlled comparison with existing U1 if repaired cheaply. **NO-GO for the tested low-cost processing on the two containers**; reopen with new boundary/thickness evidence, not another replay of the same source. |
| B: surface geometry plus material occupancy | Exact triangle UDF for the diagnostic contact reference; a separate, versioned material oracle with explicit unknown regions | Prepare the surface-query interface. **Full B remains blocked** because the material oracle is not established. Masking all difficult regions or substituting BPS does not count as resolving penetration. |

Plasticbox/trashcan each have 14 reviewed free probes and 30 unknown probes,
with **no independently established material-interior positives**. For trashcan,
the tested winding and ray methods classify all seven reviewed cavity-air probes
as material. Smalltable passes nine material and ten free local probes, but is
not a whole-mesh self-intersection/solid certificate. This is a bounded failure
of tested methods, not proof that all repairs are impossible.

The material probe has already run on both hosts: classifications match, maximum
distance difference is 0 m, winding difference is 4.44e-16. AutoDL used 348.07 s
and 145.48 MiB peak RSS. Do not repeat it without new geometry or a changed oracle.
The previously completed BPS audit also does not need repeating.

The subsequent available-upstream comparison found identical face indices and
vertex correspondence between cleaned OBJ and canonical PLY. Using the saved
rotation/translation and a fitted positive uniform scale, maximum vertex
residuals are 7.27e-8 m (plasticbox) and 8.72e-8 m (trashcan). Sections agree;
the available cleaned OBJ does not restore the missing second material surface.
The fitted scale is not a replay of historical sequence preprocessing.

Local evidence: `outputs/upstream_canonical_audit_20260908/report.md`,
`outputs/material_oracle_probe_autodl_20260908_v1/cross_host_verification.json`,
and the frozen `tests/fixtures/material_oracle_probes_v1.json`. Outputs are local
artifacts, not assumed to be included by Git synchronization.

## Bounded check for alternative geometry

The inspected `captured_objects` directory supplies cleaned/simplified OBJ files;
no unsimplified alternatives or measured wall thickness for the two containers
were established. The additional `.ply.obj` files under the SDF directory are
not an independent surface reference: `compute_rest_pose_object_sdf.py` and
`compute_obj_sdf.py` call marching cubes on the SDF and export the zero level set.
They must not be used to validate a repair of that same SDF.

The official [OMOMO README](https://github.com/lijiaman/omomo_release#testing)
and [CHOIS README](https://github.com/lijiaman/chois_release#prerequisites)
advertise their dataset downloads. The inspected pages do not establish a
separate raw-scan/thickness source. The contents of any additional, uninspected
archive remain **UNVERIFIED**; this is not a claim that such evidence cannot exist.
No additional large archive was downloaded.

## Minimal engineering choice

Keep A as the original U1 definition, but stop spending repeated audit runs on
unchanged container inputs. The smallest forward step now is unsigned surface
query correctness, with the material-oracle dependency explicitly unresolved.
Use the complete canonical triangle surface as the computational distance
reference. It is exact distance to the provided triangles, not certified distance
to the physical object's material boundary. Denser surface samples can later be
an acceleration approximation validated against this reference; BPS remains a
coarse unsigned baseline. Neither adds missing inside/outside semantics.

B is an engineering separation, not an established novel method. If its sign
and distance share the same reliable material boundary, it can reproduce a
signed-distance objective. A hard binary occupancy indicator alone supplies
no useful gradient away from its discontinuity; a future penetration term must
specify and independently test its gradient/exit direction. Confidence inferred
only from oracle agreement is not a calibrated semantic confidence measure.

A contact-only ablation could answer a narrower question with matched U0 data,
checkpoint, split and contact supervision. It must be named and reported as
contact-only, cannot claim penetration improvement from the old evaluator, and
cannot silently replace the frozen contact-plus-penetration U1. No such training
change is implemented or approved by this document.

This surface interface can be reused by a future U3 canonical surface bank for
distance/coverage checks. It does not implement predicted key selection, a
network or any U3/U4 mechanism. Whether local reliable geometry improves HOI is
a later empirical hypothesis; the present CPU tests cannot establish it.

## Finite acceptance sequence

| Gate | Required evidence | Current state |
|---|---|---|
| G1: material semantics and evaluation | Versioned boundary/component/unknown policy; independent material/free probes including container walls, bottom, cavity and rim; no failures on reviewed known probes; unknown reported separately. Training and evaluation must use the same material definition and frozen geometry version. | **BLOCKED** for both containers; legacy evaluation still reads faulty source SDF. |
| G2: numerical and integration correctness | Distance/exit/occupancy correctness plus positional/pose gradients; query units/axes/transforms/OOB verified; source integrity before cache fidelity. Meaningful CPU regression tests and strict baseline model/EMA loading evidence under the frozen protocol. | Existing 14 CPU tests and shell checks PASS (prior evidence). Cache replay PASS for the two objects. New unsigned derivative probe is only one subcheck, not full G2. |
| G3: target runtime | After G1/G2 and explicit GPU authorization: existing CUDA and frozen 300-step smoke, finite losses/gradients, expected parameters updated, frozen split/checkpoint, measured peak memory. | **BLOCKED / NOT RUN**. |

Only after these gates pass should the originally specified 20k U1 run be
considered. A three-object prototype cannot certify the remaining training
objects. Before a full-object run, every included object must satisfy the same
geometry criteria; exclusions require a declared matched U0/U1 subset experiment.
There is no need to finish U2/U3/U4 or settle novelty before this diagnostic U1.

Source integrity is a prerequisite to 64/256 fidelity. MAE/P95/sign agreement
against an untrusted source only measure reproduction of that source. Preserve
the audit as a regression/approximation check; do not use its PASS to certify
physical correctness. Do not generate a full 256³ field before G1 is established.

The next material-dependent operation needs an independently supported boundary
or thickness input for plasticbox/trashcan, bound to its provenance and reviewed
sections. Without it, make a deliberate experiment-scope decision rather than
auto-thickening walls, capping the top, or calling unknown regions free.

## Independent CPU derivative diagnostic

`scripts/probe_surface_query_gradients.py` uses the first frozen probe from each
of four distinct regions per object (12 points total), without selecting by
observed derivative results. It checks query-position, object-translation and
local SO(3) derivatives at two finite-difference scales. Each perturbation runs
the full point-to-triangle search. It also tests a small surface-directed step.

Two analytic smooth queries test face/edge distances. A separate medial-axis
fixture must explicitly fail differentiability; unsigned distance is not smooth
everywhere. The nearest-point envelope derivative is a local first derivative,
not a claim about globally smooth gradients, Hessians, material occupancy,
penetration, GT contact calibration or end-to-end model backpropagation.

Local execution completed on 2026-09-08, rechecked on 2026-09-09 against all six
bound input hashes. Final artifact: `outputs/surface_query_gradients_20260908_v2/summary.json`.

| Object | Fixed probes passing | Maximum scaled derivative error |
|---|---:|---:|
| plasticbox | 4/4 | 2.05e-10 |
| trashcan | 4/4 | 5.96e-8 |
| smalltable | 4/4 | 4.58e-8 |

Errors are absolute differences between autodiff and central finite differences;
angular derivatives (m/rad) are divided by object max extent. They are not SDF
distance errors. Acceptance tolerance was 5e-5 at both fixed step scales.
All sampled surface-directed steps reduced UDF. Runtime was 16.67 s, peak
Windows process working set 189.86 MiB. The analytic medial fixture correctly
reported non-differentiability. Both focused unit tests PASS, including existing
output/data-root guards that preserve sentinel files. `git diff --check` passed.
No model-gradient or target-server execution is implied by these local results.

The current result justifies an isolated unsigned-query backend experiment if
needed. It does not certify the physical surface or supply a material label.
In particular, gradients evaluated at an `unknown` probe do not change that
probe's semantic label.

The next unique safe AutoDL operation, **after syncing this new script**, is:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B scripts/probe_surface_query_gradients.py \
  --data_root_folder /root/autodl-tmp/mamihoi/data/processed_data \
  --probe_manifest tests/fixtures/material_oracle_probes_v1.json \
  --output_dir /root/autodl-tmp/mamihoi/outputs/sdf_gate0/stage_a/surface_query_gradients_v1
```

The command refuses an existing output directory and writes outside processed
data. It checks mesh/manifest/script/helper hashes before and after; no SDF,
cache, motion dataset, BPS tensor or trainer is loaded. It uses existing CPU
NumPy/PyTorch, batch 8 and triangle chunks of 1024, without native compilation.
