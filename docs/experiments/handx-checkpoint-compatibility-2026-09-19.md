# HandX Public Checkpoint Compatibility Diagnostic

Date: 2026-09-19

## Scope

Determine whether the public HandX `layers12` diffusion checkpoint can run as
a frozen sequence-level hand prior and whether real MaMi wrist trajectories can
be injected without changing the checkpoint.

Raw metrics:

```text
docs/experiments/handx_compat_artifacts/*.json
```

## Confirmed Repository Facts

- `alexzhang598/HandX-diffusion/layers12/model.pt` is public and is documented
  as the paper-best diffusion checkpoint.
- The checkpoint SHA-256 is
  `9247465c2277d85c40ec6de649dd26f7a94eccfc3ab356c451cdc1c32208e335`.
- The checkpoint omits the frozen T5 encoder. The missing T5 weights are
  restored from `google-t5/t5-base`.
- `run_wrist_traj.py` fixes wrist joints `0` and `21` for all 60 frames using
  inference-time spatial/temporal masking. No separate wrist-conditioned
  checkpoint is required.
- HandX conditions on text, not object geometry. Its public checkpoint has
  `contact_prediction: false` and `contact_loss: false`.
- No public autoregressive checkpoint was found in the Hugging Face model repo
  or GitHub Releases.

## Environment and Data

The server environment used:

```text
NumPy 1.23.5
Torch 2.1.2+cu118
Transformers 4.44.2
```

The NumPy-2 archive issue is handled by a local HandX-venv compatibility shim
that aliases `numpy._core` only while an NPZ member is unpickled.

Training normalization was computed on the server:

```text
frames: 423,645
mean/std shape: (42, 4)
```

## Official Wrist-Mask Smoke

The official validation sample ran with the public checkpoint:

```text
denoising iterations: 1000
runtime:               about 98 seconds
sequence length:       60
```

Results:

| Metric | Value |
|---|---:|
| Wrist RMSE | 2.369 mm |
| Wrist mean error | 2.355 mm |
| Speed ratio vs GT | 0.938 |
| Acceleration ratio vs GT | 0.724 |
| Jerk ratio vs GT | 0.656 |

The checkpoints and inference-time wrist masking work. Temporal velocities are
not amplified relative to the ground-truth motion.

## MaMi Wrist Mapping

The actual MaMi 24-joint ordering is not the standard SMPL wrist ordering used
in the first attempt. Hand-mesh centroid matching identified:

```text
left wrist:  joint 20
left palm:   joint 22
right wrist: joint 21
right palm:  joint 23
```

The first MaMi run using joints `10/14` was invalid and is preserved only as a
debugging record.

After correcting the wrist indices and tensor ordering, a hard mask tracks the
injected MaMi wrists to numerical precision:

```text
monitor wrist RMSE:      1.21e-5 mm
clothesstand wrist RMSE: 7.71e-6 mm
```

Therefore the wrist injection and normalization path is technically correct.

## Generated Skeleton Consistency

The generated motion is the HandX `joint_pos_w_scalar_rot` representation.
Bone-length variability is used as a structure diagnostic.

| Source | Left bone CV | Right bone CV |
|---|---:|---:|
| HandX ground truth | 1.14% | 1.20% |
| Official HandX wrist smoke | 5.61% | 6.20% |
| MaMi monitor, raw wrist geometry | 16.47% | 21.47% |
| MaMi clothesstand, in-range wrist distance | 14.84% | 16.81% |

The first MaMi case (`monitor`) has a mean left-to-right wrist distance around
`1.22 m`, far outside the HandX training distribution whose median is about
`0.22 m` and 99th percentile about `0.49 m`.

Selecting `sub16_clothesstand_017` with mean wrist distance about `0.21 m`
improves the input distribution but does not solve the generated-skeleton
deformation. Even in-range MaMi wrists produce about 15-17% bone-length
variation.

## Decision

```text
Public HandX checkpoint availability: GO
Official wrist-conditioned inference: PASS
Direct raw MaMi wrist -> HandX integration: CONDITIONAL NO-GO
```

The checkpoint is usable and should be retained. Raw MaMi wrist positions alone
are not a sufficient HandX conditioning contract. The generated representation
must be converted or projected to a valid hand skeleton/MANO state before it
can serve as a finger prior.

HandX still cannot guarantee object contact because it receives no object
geometry. ContactOpt remains necessary as an object-aware local correction
stage.

## Required Follow-Up

The next diagnostic must add one of:

1. a learned MaMi-to-HandX wrist/adaptor canonicalizer; or
2. an explicit skeleton/manifold projection after HandX generation; or
3. a small fine-tuned adapter while keeping the HandX diffusion backbone
   frozen.

No world-model training is justified until the generated hand state passes a
valid bone-length and MANO-reconstruction gate on multiple subjects and
objects.
