# CPU material-oracle probe: local evidence and decision

The standalone `scripts/material_oracle_probe.py` is implemented and locally
run for plasticbox, trashcan and smalltable. It does not import a trainer,
construct a dataset, invoke Torch/CUDA, read old SDF/BPS, export repaired meshes,
or generate a voxel grid. Existing inputs are never written. Results require a
new directory, outside the input data root. No packages were installed or C++
compiled. Existing training, loss, dataset and evaluator code is unchanged.

## Evidence protocol

The label manifest is `tests/fixtures/material_oracle_probes_v1.json`. Its 112
points were fixed after viewing raw-mesh sections and before occupancy results.
Every point records coordinates, region, label, source and applicable evidence
checks. Mesh SHA is mandatory. These are local geometry-reviewed labels, not
measured wall thickness or globally certified physical geometry.

Six raw sections per object and axial intersection coordinates are in
`outputs/material_geometry_review_20260907`. The reviewed open U-shaped
vertical sections, perimeter-only horizontal sections, distance from walls,
and unobstructed upward corridor support seven cavity-air points/container.
A central coordinate alone never supplies a label. Strictly exterior AABB
points and above-opening points provide additional free-space checks.

For trashcan, middle-height lines cross one sheet at each side, and the central
vertical line crosses one bottom sheet. No independently supported material
thickness is established at these locations. Both sides of the observed wall
and bottom, plus upper-wall/rim probes, are therefore **unknown**. This covers
the required wall/bottom material questions without inventing positive labels.
Plasticbox has similar missing/ambiguous thickness evidence plus extensive
overlap. Smalltable's tabletop and leg sections have paired boundaries;
nine points inside those reviewed regions are material. Its small reverse
component and on-boundary points remain unknown.

Material accuracy with zero supported material probes is `null` / NOT
ASSESSABLE. Unknown-labelled points never count as correct material/free.
The report distinguishes actual misclassifications from abstentions, includes
denominators, and reports uncertainty coverage. The optional algorithm
agreement filter is explicitly NOT calibrated semantic confidence.

## Algorithms and bounded processing

Winding is the direct sum of signed triangle solid angles divided by 4 pi;
thresholds 0.25/0.5/0.75 are reported separately. This is generalized winding,
not an implementation of an accelerated fast-winding approximation. It uses
NumPy float64 blocks of eight points by 1024 triangles. The established method
also appears in [libigl's winding implementation](https://github.com/libigl/libigl/blob/main/include/igl/winding_number.cpp).

The independent ray implementation uses Moller-Trumbore intersections along
seven fixed oblique directions. Coincident distances within 1e-7 times the
object extent are deduplicated; naive face-hit parity is retained separately
to expose reverse-overlap/shared-edge effects. Missing boundaries are not
repaired by this deduplication. Winding is tested at six axial perturbations
of 1e-4 extent, parity at a fixed oblique perturbation of comparable size.
Exact on-surface occupancy is not treated as binary truth.

The only candidate processing, entirely in memory, is exact-position vertex
welding, exactly zero-area removal, unordered face deduplication, and coherent
orientation across manifold edges. Closed orientable components are oriented
outward as an explicitly named **candidate**. No vertex moves, holes, caps,
thickness or surfaces are invented. Open/nonmanifold patches have no certified
global outward orientation. Component sum, largest component, outward union,
and global orientation sensitivity are recorded rather than selected as truth.

Exact point-to-triangle unsigned distance reuses the tested mathematical
primitive from the BPS diagnostic; no BPS query is run. Distances on real meshes
measure their triangle union, not unknown physical surface accuracy. A material
point's closest-boundary distance is checked against the first intersection
along its nearest-exit direction and the inside-to-free transition. This is a
local consistency check on reviewed material, not a global solid certificate.

Reliable self-intersection certification is **NOT RUN**: there is no installed
reliable geometry engine in this local environment. Face counts, orientation
and closedness cannot substitute for it. This limit explicitly blocks a full
smalltable A GO. No backend was installed to conceal that limitation.

## Analytic validation

Six focused tests PASS. Independent box and thick open-cup definitions check
327 points, using analytic occupancy and distances to axis-aligned rectangles
instead of the triangle-query implementation as truth. Winding and all seven
ray directions classify every fixture point correctly. Maximum distance
error is 1.11e-16 m; maximum exit consistency error is 1.39e-17 m.

The cup explicitly has occupied walls/bottom, a free cavity and no top cap.
Other tests cover reverse duplicate cancellation, coincident-hit deduplication,
open-boundary preservation, rigid transforms/scaling and unknown/empty metrics.

## Real-object results

Final local artifacts: `outputs/material_oracle_probe_20260907_v2/summary.json`,
three `<object>_detail.json` files, and three section PNGs. Earlier v1 artifacts
are preserved; v2 separates abstentions from errors and adds memory/per-label
stability metadata. Both runs produce the same geometric conclusions.

| Object | Material / free / unknown probes | Material accuracy, winding / majority parity | Free accuracy, winding / majority parity | Semantic unknown coverage | Decision |
|---|---:|---:|---:|---:|---|
| plasticbox | 0 / 14 / 30 | NOT ASSESSABLE / NOT ASSESSABLE | 14/14 / 10/14 | 68.2% | A NO-GO for tested finite processing |
| trashcan | 0 / 14 / 30 | NOT ASSESSABLE / NOT ASSESSABLE | 7/14 / 7/14 | 68.2% | A NO-GO for tested finite processing |
| smalltable | 9 / 10 / 5 | 9/9 / 9/9 | 10/10 / 10/10 | 20.8% | Local A feasibility GO; whole-object Gate BLOCKED |

These accuracy values are identical before/after the candidate processing at
winding threshold 0.5. Unknown coverage is the fraction of deliberately
selected probes whose semantics remain unknown, not a percentage of object
volume/surface. It must not be read as a dataset-wide estimate.

Plasticbox's raw cavity winding is about 0.0001-0.0007. Opposite-face
cancellation can produce this apparent free-space success without defining
any material volume. After limited processing, cavity winding is about
-0.60 to -0.44: open-patch orientation remains ambiguous. Majority parity
misclassifies `cavity_0`, `cavity_1`, `cavity_3`, `cavity_6`. Coordinates,
individual ray votes and label sources are retained in its detail report.

Trashcan's seven reviewed cavity-air probes are **all** classified material
by both winding and majority parity. Candidate winding there is approximately
0.769-0.939; changing the tested threshold does not repair the cavity meaning.
Upward-facing rays escape the opening while most other rays hit a wall/bottom.
This is evidence of wrong material semantics despite method agreement.

| Object | Candidate boundary / nonmanifold edges | Ray-direction-sensitive probes | Threshold-sensitive probes | Position-sensitive winding / ray probes |
|---|---:|---:|---:|---:|
| plasticbox | 481 / 6,397 | 41/44 | 9/44 | 2/44 / 7/44 |
| trashcan | 775 / 802 | 32/44 | 2/44 | 6/44 / 6/44 |
| smalltable | 0 / 0 | 2/24 | 0/24 | 2/24 / 2/24 |

Smalltable's instabilities occur only at unknown/on-boundary probes. Its nine
reviewed material points have valid local exits, with maximum independent
ray-versus-closest distance difference 1.13e-17 m. Plasticbox/trashcan exit
distance is NOT ASSESSABLE because trusted material-interior labels are absent;
no UDF value is relabelled penetration depth. Changing the smalltable tiny
component orientation changes three unknown probe labels, demonstrating why
outward component union must not silently define physical truth.

Raw-versus-candidate UDF changes at all probes are zero. The largest residual
from sampled raw face interiors to candidate triangles is 3.89e-17 m across
objects. This certifies sampled numerical surface preservation, **not**
physical geometry, sign correctness, or exhaustive Hausdorff distance.

Final local execution took 74.21 seconds; Windows process peak working set was
148,492,288 bytes (141.61 MiB). Each mesh's SHA was verified before and after.
No AutoDL result is claimed for this new probe; prior BPS AutoDL evidence is
not reused as material-probe execution evidence.

## Decision and limits

Do not regenerate plasticbox/trashcan signed fields from this candidate. The
tested finite processing leaves unresolved material thickness and substantial
topological defects. Filling the opening or arbitrarily thickening sheets
would introduce a new material model, not recover verified ground truth.
Additional original geometry/thickness evidence is required. The amount of
manual modelling needed is unknown; this pilot does not prove every possible
low-cost repair impossible or warrant a universal A impossibility claim.

Smalltable supports continued local A investigation without changing its
observed surfaces, but its tiny component semantics and global self-intersection
status still prevent a whole-object signed-field Gate PASS. Sparse local
success does not authorize training on all objects.

Reusable B interfaces are exact surface closest-point/distance, query and
region manifests, independent occupancy diagnostics, and explicit unknown
reporting. No B loss or learned field is implemented. An unknown material
region cannot supply penetration supervision through those interfaces.

## Evaluator dependency and future version contract

`train/trainer_control_GAPA_chois.py:521` loads seen-object geometry directly
from `rest_object_sdf_256_npy_files/<object>.ply.npy` and JSON.
`compute_hand_penetration_metric()` at line 1501 transforms hand/body vertices
to canonical coordinates, calls that loader, then samples signed distance.
It therefore remains tied to the old sign-integrity failure even if a future
training cache is repaired. This code was inspected, not imported or changed.

Training and evaluation must share the same versioned **material definition**:
original mesh hashes, candidate recipe and boundary hashes, component/occupancy
policy, reviewed labels, unknown policy, canonical transform, units, axes,
normalization, query bounds and backend/version/tolerances. Record these under
one geometry-oracle manifest ID in both cache metadata and result provenance.
Training/evaluation may use different resolutions/backends only after their
error against that same definition is checked. Independently implemented
evaluation prevents shared numerical bugs; it must not use a different solid.

Re-evaluate U0 and U1 under the same frozen evaluation geometry and metric
version. Do not compare a repaired U1 evaluator against old-SDF U0 scores.
The legacy mean-negative-SDF score must not be renamed penetration ratio or
conditional depth. No new metric or evaluator change is made here.

README/ Gate runbook fact updates needed (listed only, not edited): mark U1
Implemented/BLOCKED with source-sign and evaluation-geometry blockers; retain
cache provenance/replay PASS and BPS coarse-baseline CPU PASS as separate facts;
add this local material-probe result with NOT ASSESSABLE fields; distinguish
implemented SO(3)/OOB handling from pending execution evidence; move material
semantics/source integrity ahead of 64/256 fidelity. CUDA smoke and U1 20k
remain BLOCKED. No novelty or U2/U3/U4 status change follows from this result.

## Reproduction and next action

After copying the new script and probe manifest into an unchanged b1fb418
checkout (the existing triangle-helper file is already present), CPU replay is:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B scripts/material_oracle_probe.py \
  --data_root_folder /root/autodl-tmp/mamihoi/data/processed_data \
  --probe_manifest tests/fixtures/material_oracle_probes_v1.json \
  --objects plasticbox trashcan smalltable \
  --output_dir /root/autodl-tmp/mamihoi/outputs/sdf_gate0/stage_a/material_oracle_probe_v1
```

This needs existing NumPy/matplotlib only, does not load motion joblib, and
does not need the earlier BPS file-descriptor adjustment. Existing output
directories fail closed. The optional `--review_only` replaces the manifest
argument to regenerate visual review artifacts in a separate new directory.

The recommended next substantive operation is a **read-only upstream-geometry
comparison** for plasticbox/trashcan: inspect available original/cleaned meshes
and the canonicalization mapping to determine whether a second material surface
or thickness evidence exists upstream. Do not choose a thickness or cap a
cavity before that evidence is found. AutoDL replay of this diagnostic is a
reproducibility option, not a way to resolve missing material semantics.

Confirmed available inputs for that next operation are
`captured_objects/plasticbox_cleaned_simplified.obj` and
`captured_objects/trashcan_cleaned_simplified.obj`, together with existing
`rest_object_geo/<object>.json` rest-pose metadata. Their geometry has not been
compared in this probe run. No extra input is presumed to contain thickness.
