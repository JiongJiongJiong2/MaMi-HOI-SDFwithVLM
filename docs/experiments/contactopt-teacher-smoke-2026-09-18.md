# ContactOpt Teacher Smoke

Date: 2026-09-18

## Scope

Test whether the official ContactOpt pipeline can refine an existing MANO
hand mesh toward an object mesh on the clean GPU server, before attempting any
MaMi integration or finger world-model training.

This is a single-sample dependency and behavior smoke, not evidence that
ContactOpt preserves a generated MaMi wrist trajectory or object interaction
state.

## Environment

- ContactOpt: `9eeb59a1cdddf4a5e94fec39d77808ddd5ed512c`
- manopth: `4f1dcad1201ff1bfca6e065a85f0e3456e1aa32b`
- Isolated venv: `/root/autodl-tmp/external/contactopt-venv2`
- Python 3.10.21
- NumPy 1.23.5
- Torch 2.1.2+cu118
- PyTorch3D 0.7.9
- Open3D 0.20.0
- torch-geometric 2.3.1
- torch-cluster 1.6.3+pt21cu118

The historical repository required three compatibility repairs:

1. Import `PointConv` from `torch_geometric.nn.conv`.
2. Copy `hand_contact` in `HandObject.load_from_ho`.
3. Install the system `libegl1` and `libgl1` libraries required by Open3D.

The MANO left and right model files were linked from the existing MaMi
assets into `ContactOpt/mano/models`.

## Command

```bash
cd /root/autodl-tmp/external/ContactOpt
OMP_NUM_THREADS=8 PYTHONPATH=. \
  env -u http_proxy -u https_proxy \
  /root/autodl-tmp/external/contactopt-venv2/bin/python \
  contactopt/run_user_demo.py
```

The official demo completed in 64.23 seconds and wrote
`data/optimized_user.pkl`.

The repository-side evaluator was run against that output:

```bash
OMP_NUM_THREADS=8 PYTHONPATH=. \
  env -u http_proxy -u https_proxy \
  /root/autodl-tmp/external/contactopt-venv2/bin/python \
  evaluate_contactopt_smoke.py data/optimized_user.pkl
```

## Result

The input and output hand-object states were evaluated with ContactOpt's
capsule-contact model and with a direct hand-vertex to object-vertex nearest
distance.

| Metric | Input | Refined | Change |
|---|---:|---:|---:|
| Mean nearest hand-to-object vertex distance | 17.166 mm | 12.336 mm | -28.13% |
| Mean ContactOpt hand contact weight | 0.18736 | 0.29908 | +59.62% |
| Mean ContactOpt object contact weight | 0.08779 | 0.10157 | +15.70% |
| Mean unaligned hand-joint movement from input | 0 mm | 27.64 mm | non-trivial refinement |

The demo's ground-truth object is intentionally a copy of the input object, so
the reported output alignment error measures how far refinement moved the
hand, not a real held-out accuracy.

## Decision

GO for ContactOpt as a potential contact-refinement teacher on an existing
hand-object mesh pair.

NO-GO for claiming that this smoke solves finger world modeling. ContactOpt
does not generate temporal finger dynamics, infer object motion, or guarantee
preservation of the MaMi wrist and object trajectories.

## Next Gate

Convert one actual MaMi-generated hand-object pair into the ContactOpt
MANO-PCA input contract, run refinement with the wrist and object frame
frozen, and verify contact improvement without an unacceptable wrist or
object-frame drift.
