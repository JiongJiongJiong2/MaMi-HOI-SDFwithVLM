"""Small CPU-only unsigned triangle-query derivative diagnostic, not a loss.

Uses frozen geometry probes, never motion/dataset/SDF/BPS or trainer entrypoints.
Nearest-point envelope gradients are valid away from zero distance and medial
ties. Finite differences re-run the full triangle search; no fixed-face oracle.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path
import sys
import time

os.environ['CUDA_VISIBLE_DEVICES'] = ''
for key in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
    os.environ[key] = '1'
import numpy as np
import torch
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.audit_bps_surface_queries import nearest_triangles, read_canonical_ply, sha256, memory_snapshot

torch.set_num_threads(1)
DTYPE = torch.float64
OBJECTS = ('plasticbox', 'trashcan', 'smalltable')


def skew(w):
    z = w[0] * 0
    return torch.stack((z, -w[2], w[1], w[2], z, -w[0], -w[1], w[0], z)).reshape(3, 3)


def canonical(state, rotation):
    return (state[:3] - state[3:6]) @ (rotation @ torch.matrix_exp(skew(state[6:])))


def check_point(q, triangles, extent):
    rotation = torch.matrix_exp(skew(torch.tensor([.23, -.17, .31], dtype=DTYPE)))
    com = np.array([.37, -.21, .46])
    state = np.r_[q @ rotation.numpy().T + com, com, np.zeros(3)]
    variable = torch.tensor(state, dtype=DTYPE, requires_grad=True)
    local = canonical(variable, rotation)
    d, cp = nearest_triangles(np.asarray(q)[None], triangles, query_batch=8)
    if d[0] <= extent * 1e-9:
        return dict(status='ZERO_DISTANCE_NONDIFFERENTIABLE', distance_m=float(d[0]))
    distance = torch.linalg.vector_norm(local - torch.tensor(cp[0], dtype=DTYPE))
    gradient = torch.autograd.grad(distance, variable)[0].numpy()
    # Angular derivatives have units m/rad; divide by extent to compare scales.
    units = np.r_[np.ones(6), np.full(3, extent)]
    errors = []
    finite_gradients = []
    for relative_step in (1e-5, 2.5e-6):
        steps = relative_step * np.r_[np.full(6, extent), np.ones(3)]
        perturbed = []
        for axis in range(9):
            for sign in (1, -1):
                s = state.copy()
                s[axis] += sign * steps[axis]
                perturbed.append(canonical(torch.tensor(s, dtype=DTYPE), rotation).numpy())
        ds, _ = nearest_triangles(np.array(perturbed), triangles, query_batch=8)
        fd = (ds[::2] - ds[1::2]) / (2 * steps)
        finite_gradients.append(fd.tolist())
        errors.append((np.abs(fd - gradient) / units).tolist())
    # Test one small surface-directed step; it makes no inside/outside claim.
    direction = (q - cp[0]) / d[0]
    step = min(extent * 1e-5, float(d[0]) / 4)
    moved_d, _ = nearest_triangles((q - step * direction)[None], triangles, query_batch=8)
    passed = max(max(e) for e in errors) < 5e-5 and moved_d[0] < d[0]
    return dict(status='PASS_SAMPLED_DERIVATIVE' if passed else 'REVIEW_NONSMOOTH_OR_ERROR',
                distance_m=float(d[0]), closest_point=cp[0].tolist(),
                gradient_hand_com_rotation=gradient.tolist(),
                finite_difference_gradients=finite_gradients,
                scaled_absolute_errors=errors, max_scaled_error=max(max(e) for e in errors),
                distance_after_step_m=float(moved_d[0]), step_m=step,
                canonical_roundtrip_error_m=float(np.max(np.abs(local.detach().numpy() - q))))


def fixtures():
    triangle = np.array([[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]]])
    reports = []
    for q, expected in (([.2, .2, .3], .3), ([.6, -.2, .3], np.sqrt(.13))):
        result = check_point(np.array(q), triangle, 1.)
        assert result['status'] == 'PASS_SAMPLED_DERIVATIVE', result
        assert abs(result['distance_m'] - expected) < 1e-12
        reports.append(result)
    # Explicit medial tie: d(z)=1-|z|. A nearest-point branch has a derivative
    # but the unsigned distance does not; symmetric differences must detect it.
    plane = np.array([[[-2., -2., 0.], [2., -2., 0.], [0., 2., 0.]]])
    pair = np.concatenate((plane + [0, 0, 1], plane - [0, 0, 1]))
    tie = check_point(np.zeros(3), pair, 2.)
    assert tie['status'] == 'REVIEW_NONSMOOTH_OR_ERROR'
    return dict(smooth=reports, expected_medial_nondifferentiability=tie)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data_root_folder', type=Path, required=True)
    parser.add_argument('--probe_manifest', type=Path, required=True)
    parser.add_argument('--output_dir', type=Path, required=True)
    args = parser.parse_args()
    root, out = args.data_root_folder.resolve(), args.output_dir.resolve()
    if out == root or root in out.parents:
        raise ValueError('Diagnostic output must be outside the data root')
    if out.exists():
        raise FileExistsError(out)
    manifest = json.loads(args.probe_manifest.read_text(encoding='utf-8'))
    paths = [args.probe_manifest, Path(__file__), Path(__file__).with_name('audit_bps_surface_queries.py')]
    paths += [root / 'rest_object_geo' / f'{name}.ply' for name in OBJECTS]
    before = {str(p): sha256(p) for p in paths}
    for name in OBJECTS:
        p = root / 'rest_object_geo' / f'{name}.ply'
        assert before[str(p)] == manifest['objects'][name]['mesh_sha256'], name
    out.mkdir(parents=True, exist_ok=False)
    start = time.perf_counter()
    report = dict(scope='UNSIGNED_QUERY_DERIVATIVE_ONLY_U1_BLOCKED',
                  fixtures=fixtures(), numpy=np.__version__, torch=torch.__version__, objects={})
    for name in OBJECTS:
        v, f = read_canonical_ply(root / 'rest_object_geo' / f'{name}.ply')
        extent = float(np.ptp(v, axis=0).max())
        # Fixed manifest order, first point of each of four distinct regions.
        # Selection precedes derivative calculation; every outcome is retained.
        selected, regions = [], set()
        for probe in manifest['objects'][name]['probes']:
            if probe['region'] not in regions and len(selected) < 4:
                selected.append(probe)
                regions.add(probe['region'])
        results = []
        for probe in selected:
            result = check_point(np.array(probe['point']), v[f], extent)
            results.append(dict(probe=probe, **result))
        report['objects'][name] = results
        print(name, [(p['probe']['id'], p['status']) for p in results], flush=True)
    after = {str(p): sha256(p) for p in paths}
    assert before == after, 'Input hash changed'
    report.update(input_hashes=before, input_hashes_unchanged=True,
                  elapsed_seconds=time.perf_counter() - start, memory=memory_snapshot())
    (out / 'summary.json').write_text(json.dumps(report, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Written', out / 'summary.json', flush=True)


if __name__ == '__main__':
    main()
