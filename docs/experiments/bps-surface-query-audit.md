# BPS / mesh unsigned surface-query diagnostic

This is a small CPU-only geometry comparison, not a training experiment or a
release of the U1 signed-loss Gate. Existing source SDF signs remain untrusted.
The diagnostic never reads/writes SDF fields, repairs meshes, constructs the
dataset, runs CUDA, or changes training code. Only a new output directory is
created; an existing output directory is rejected.

## Inputs and query convention

`bps.pt['obj'] + rest_object_geo/<object>.npy[0]` restores 1024 BPS anchors.
The repository encoded these deltas from canonical mesh **vertices**. The
script verifies every restored anchor against the current canonical vertices;
1024 anchors need not represent 1024 distinct vertices. This measures nearest
restored-anchor querying, not the trained model's use of its BPS conditioning.

Three methods query the same points: nearest restored BPS anchor, nearest
full-mesh vertex, and nearest point on the full triangle union. The triangle
reference exhaustively minimizes over face interiors and closed edges in
float64, including degenerate faces. It does not require oriented or watertight
faces for unsigned distance, but does not certify that the mesh is the true
physical surface. Original vertices/faces are unchanged.

Real palm proxies are joints 22/23 from the existing 120-frame window file:
`q_canonical = (q_window_world - window_obj_com_pos) @ obj_rot_mat`.
Saved window rotations already incorporate the rest-frame adjustment.
Only sequences listed under `validation_sequences` in the frozen manifest
contribute queries. Test sequence motion is not used in metrics; joblib mmap
loads the container's metadata. No new split is selected. Contact strata use
the first two columns of the existing per-sequence contact annotations.
These annotations do not guarantee that a palm proxy lies on the surface.

Defaults: four objects (plasticbox, trashcan, largebox, smalltable), seed 1,
four validation sequences/object, at most three samples per hand/contact stratum
per sequence, frame stride 5. Add paired palm perturbations at one `h64`,
64 area-weighted surface samples, 128 paired near-surface offsets, 64 uniform
cube probes, and 27 central-volume probes for plasticbox/trashcan.
`h64 = max(mesh bounds extents)/63` is a physical comparison scale, not an
SDF query. Surface sampling ignores repeated vertex-index triplets **only in
the sampling distribution**, so opposite duplicate faces do not overweight
coverage. Distance queries still use every original face.

Near-surface offsets probe both sides without labeling inside/outside.
Central-volume probes are not verified cavity labels. Thin walls and actual
bucket-cavity coverage therefore require subsequent visualization/region
annotation before any region-specific claim. The present samples are a
screening pilot, not exhaustive surface coverage.

## AutoDL command

Copy `scripts/audit_bps_surface_queries.py` into the same path in the existing
server checkout first. No package installation or source/cache regeneration is
needed when its existing NumPy, CPU-loadable PyTorch, and joblib are available.
Run from `/root/MaMi-HOI-SDFwithVLM` in the existing `mami_hoi` environment:

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B scripts/audit_bps_surface_queries.py \
  --data_root_folder /root/autodl-tmp/mamihoi/data/processed_data \
  --split_manifest /root/autodl-tmp/mamihoi/outputs/protocol/split_seed1.json \
  --output_dir /root/autodl-tmp/mamihoi/outputs/sdf_gate0/stage_a/bps_surface_queries_20260907_v1 \
  --objects plasticbox trashcan largebox smalltable \
  --seed 1 --query_batch_size 16 --triangle_chunk_size 1024
```

The manifest in the supplied server log has SHA-256
`665aeea1d076ec6ddf6605018511b4ee46f1ef9dd84ed4c09922a5d0187fe0db`.
Use that frozen file; do not regenerate/overwrite it. An optional
`--geometry_only` replaces `--split_manifest` when only synthetic queries are
intended; such output explicitly reports real-hand validation as NOT RUN.

Inputs are mmap/chunked. The motion file must be at most 512 MiB and its array
fields must actually be memmaps; eager fallback is rejected. Default pairwise
blocks are 16 queries by 1024 triangles/points, never a full query-by-mesh
allocation. CPU threads are fixed to one. The report includes process peak RSS
and cgroup memory readings on Linux. A low algorithmic footprint does not
guarantee headroom in a cgroup shared with other processes; server measurements
remain necessary. `memory.peak` can include prior/shared processes.

## Outputs and interpretation

`summary.json` includes input/script SHA-256, versions, selection provenance,
unique anchor counts, metrics per query stratum, memory, and CPU timings.
`queries.csv` contains the exact canonical queries, sequence/frame/joint/contact
provenance, reference closest points, distances, errors, and pull angles.

Errors are measured against triangle-union distance in mm and units of `h64`.
Coverage is the sampled surface fraction within 2/5/10 mm of anchors/vertices.
The 10 mm proximity miss rate is a geometry diagnostic; it is **not** the
repository's official 50 mm Contact-F1 metric. Pull direction means
`closest_point - query`, not a face normal or an inside/outside direction.
Zero-distance directions are excluded; medial-axis ties may have several
equally valid directions. This does not test differentiability/autograd.

Timing uses the same 32 fixed queries per object, one warmup and three repeats,
with input loading excluded. All three implementations use exhaustive NumPy
search. These measurements cannot establish deployment latency against an
optimized BVH/KD-tree or GPU implementation. Do not mix all query strata into
one headline MAE, use test data to choose methods, or infer HOI efficacy from
this diagnostic.

Analytic CPU validation (copy the test file too if running on the server):

```bash
CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \
python -B -m unittest discover -s tests -p test_bps_surface_queries.py -v
```

## Gate and research interpretation

BPS and UDF can validate unsigned-distance fidelity/coverage, not source sign.
The triangle reference shares the source mesh with BPS; agreement is not three
independent confirmations of physical geometry. Occupancy additionally needs
a defensible inside/outside definition and is not Euclidean distance.

An unsigned contact-loss baseline is a candidate subsequent experiment, not
implemented here. To isolate signedness, compare matched queries, geometry,
distance magnitude, contact masks/targets, model/checkpoint, optimizer and
loss scale, and retain a separately valid penetration evaluation. BPS versus
SDF alone also changes approximation and spatial coverage. A general unsigned
contact attraction cannot penalize penetration by itself.

Novelty remains UNVERIFIED. A future HOI-specific reliability method needs
evidence about the interaction query distribution, uncertainty calibration,
useful directions/gradients, and held-out generation/contact/penetration
outcomes. This pilot cannot decide whether signed geometry is unnecessary.
Source-integrity failure, CUDA smoke BLOCKED, and U1 20k BLOCKED remain in force.

## Local pilot result: 2026-09-07

Completed 1,636 identical cross-method queries from four objects, including
186 original palm proxies across 16 frozen validation sequences and 372 paired
perturbations. Four analytic tests passed under both NumPy 1.26.4 and 2.2.6.
Post-run checks passed for all 27 input hashes, script hash, CSV count,
validation/test separation, surface-query reference residuals below 1e-10 m,
and removal of GT contact labels from perturbed queries. Existing repository
files, SDFs and caches were not modified.

Output: `outputs/bps_surface_queries_local_20260907/{summary.json,queries.csv}`.
The ignored local protocol mirror has exactly the frozen server manifest SHA
above; it did not replace an existing manifest. Script SHA for this run:
`604b729f336d056ad2a4a8fd31c3bbf59360e19eea36c97ded25f5ec8d0bcd16`.

| Object | Distinct BPS anchor vertices | Contact-labelled palm queries | BPS distance MAE / P95, mm | Mesh-vertex distance MAE / P95, mm | BPS sampled surface coverage within 10 mm |
|---|---:|---:|---:|---:|---:|
| plasticbox | 528 | 24 | 2.60 / 6.59 | 0.15 / 0.45 | 15.6% |
| trashcan | 550 | 18 | 11.15 / 29.11 | 0.16 / 0.60 | 21.9% |
| largebox | 388 | 24 | 17.48 / 48.36 | 0.77 / 2.44 | 10.9% |
| smalltable | 466 | 24 | 7.30 / 20.50 | 0.14 / 0.40 | 25.0% |

All reconstructed anchors matched canonical vertices within 4.0e-8 m.
Full-mesh vertices covered all 64 sampled surface points/object within 10 mm;
this is a finite-sample result, not exhaustive coverage. There are only 90
contact-labelled palm queries; their median triangle distances by object are
23.52, 10.94, 7.85 and 19.31 mm. Palm annotations therefore must not be treated
as exact zero-distance supervision.

| Object | Near-surface BPS / vertex distance P95, mm | Near-surface BPS / vertex pull-angle median | Contact-palm BPS / vertex pull-angle median |
|---|---:|---:|---:|
| plasticbox | 59.69 / 2.49 | 74.83 / 21.75 deg | 15.46 / 5.24 deg |
| trashcan | 42.89 / 1.33 | 76.89 / 20.28 deg | 58.90 / 7.89 deg |
| largebox | 49.59 / 2.29 | 81.47 / 31.02 deg | 63.04 / 19.89 deg |
| smalltable | 48.41 / 1.83 | 68.96 / 24.54 deg | 40.82 / 4.60 deg |

This supports retaining restored-anchor querying as a **coarse unsigned
baseline**, not a precise reference or sign oracle. Its coverage weakness also
occurs in source-sign controls, independently of the old source-sign failure.
Full vertices approximate triangle **distance** substantially better on these
queries, while near-surface direction errors still need attention. Neither
result establishes differentiable loss quality or generation efficacy.

Windows CPU process peak working set: 225,914,880 bytes (215.45 MiB).
Geometry/benchmark/output phase: 35.81 seconds, excluding initial basis/motion
loading and manifest/input hashing. Median exhaustive-query times across the
four objects were 18.3-19.9 us/query for BPS, 239.6-346.1 us/query for vertices,
and 10.43-26.67 ms/query for triangles. These are implementation timings, not
a performance claim against an optimized mesh acceleration structure.
AutoDL CPU replay and its cgroup memory measurements are NOT RUN locally.

The local run used existing introRL-torch Python 3.10.20, NumPy 2.2.6 and
PyTorch 2.5.1, with read-only access to already-installed joblib 1.5.2 via a
process-local import path. No packages were installed. A different local
environment failed with duplicate OpenMP runtimes before object processing;
that failed run contributes no measurements and the runtime error was not
suppressed.

The next authorized action is the same four-object AutoDL CPU replay above.
Before any later training proposal, use its evidence to define a bounded
follow-up on matched near-contact queries and verified thin-wall/cavity
regions, including direction/gradient validation. Do not add a learned field,
repair meshes, or replace contact/penetration losses as part of this pilot.
