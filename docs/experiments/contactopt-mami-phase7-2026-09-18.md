# ContactOpt on Real MaMi Outputs: Phase 7

Date: 2026-09-18

Raw result:

```text
docs/experiments/contactopt-mami-phase7-2026-09-18.json
```

## Scope

Run ContactOpt on a real MaMi candidate that already stores the SMPLX right
hand mesh and object mesh, with the wrist root and object frame frozen. This
tests whether ContactOpt can act as a finger/contact correction teacher
without changing the generated interaction trajectory.

This is one sequence, one object, and one hand. It is not a claim of
generalization across objects or subjects.

## Frozen Input

```text
sequence: sub17_monitor_026
object: monitor
candidate:
/root/autodl-tmp/contact_action_20260914/
mami_l1_heldout_51_70_20260915/candidate_seed_1/
res_npz_files/chois_wo_guidance/sub17_monitor_026.npz
```

The deterministic frame selector chose:

```text
[9, 15, 21, 28, 33, 42, 49, 54, 114, 119]
```

The MaMi output stores `pred_right_hand_verts` from the SMPLX hand with zero
hand-pose channels and `pred_object_verts` in world coordinates. A neutral
MANO right hand was aligned to the saved SMPLX hand using:

1. the same gender-specific SMPLX model and betas;
2. the repository's canonical right-hand vertex IDs;
3. a correspondence Procrustes fit from canonical SMPLX to the saved hand;
4. landmark initialization and rigid ICP from MANO to canonical SMPLX.

## ContactOpt Protocol

```text
w_opt_rot = 0
w_opt_trans = 0
w_obj_rot = 0
rand_re = 0
n_iter = 250
```

Only the 15-dimensional MANO finger PCA pose was optimized. The wrist root
rotation and translation and the world-space object mesh were frozen.

## Alignment

| Metric | Result | Threshold | Status |
|---|---:|---:|---|
| Canonical SMPLX-to-saved-hand residual | 1.564 mm | <= 3 mm | PASS |
| Aligned MANO-to-saved-hand mean distance | 11.463 mm | <= 15 mm | PASS |

The second value is the main approximation in this experiment. MaMi currently
renders neutral fingers from SMPLX rather than a stored articulated hand pose;
the MANO initialization therefore differs by roughly one centimeter before
ContactOpt refinement.

## Per-Frame Result

| Frame | Input distance mm | Refined distance mm | Distance change | Hand contact change |
|---:|---:|---:|---:|---:|
| 9 | 34.747 | 23.713 | -31.76% | +124.89% |
| 15 | 20.164 | 11.852 | -41.22% | +165.48% |
| 21 | 19.843 | 12.178 | -38.63% | +153.06% |
| 28 | 20.868 | 14.993 | -28.16% | +134.89% |
| 33 | 24.845 | 15.924 | -35.91% | +122.03% |
| 42 | 29.547 | 14.109 | -52.25% | +168.64% |
| 49 | 33.937 | 18.697 | -44.91% | +86.85% |
| 54 | 46.690 | 23.336 | -50.02% | +148.83% |
| 114 | 45.406 | 25.529 | -43.78% | +256.86% |
| 119 | 42.456 | 27.357 | -35.57% | +180.46% |

## Aggregate Result

| Metric | Result | Frozen gate | Status |
|---|---:|---:|---|
| Contact-improved frames | 10/10 | >= 7/10 | PASS |
| Distance-improved frames | 10/10 | reported | PASS |
| Mean hand contact relative change | +154.20% | > 0 | PASS |
| Mean nearest-distance relative change | -40.22% | <= +10% | PASS |
| Maximum wrist-root drift | 0.0 m | <= 0.00001 m | PASS |
| Maximum object-vertex drift | `1.49e-8 m` | <= `1e-6 m` | PASS |
| Mean MANO hand-vertex motion | 18.506 mm | reported | non-trivial refinement |

Overall Phase 7 gate: PASS.

## Decision

GO for ContactOpt as a finger-pose refinement teacher on real MaMi
hand-object states, provided that:

1. the wrist root and object frame remain frozen;
2. the MANO-to-SMPLX initialization residual is reported;
3. refinement is validated on additional sequences and objects.

NO-GO for claiming that this resolves finger world modeling. MaMi still has no
stored `pose_hand`; ContactOpt only changes an aligned neutral MANO hand. It
does not infer object motion, physical forces, or the ground-truth finger
action.

## Reproduction

```bash
cd /root/autodl-tmp/external/ContactOpt
OMP_NUM_THREADS=8 \
env -u http_proxy -u https_proxy \
/root/autodl-tmp/external/contactopt-venv2/bin/python \
run_mami_contactopt_case.py \
  --candidate-npz /root/autodl-tmp/contact_action_20260914/mami_l1_heldout_51_70_20260915/candidate_seed_1/res_npz_files/chois_wo_guidance/sub17_monitor_026.npz \
  --sequence-db /root/autodl-tmp/mamihoi/data/processed_data/test_diffusion_manip_seq_joints24.p \
  --num-frames 10 \
  --min-frame-separation 5 \
  --output-tag mami_phase7_full \
  --output-json /root/autodl-tmp/external/ContactOpt/phase7_full.json
```
