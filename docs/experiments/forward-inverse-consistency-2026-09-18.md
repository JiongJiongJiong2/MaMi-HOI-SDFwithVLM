# Forward-Inverse Consistency Diagnostic

Date: 2026-09-18

Status: complete with NO-GO

## Protocol

The diagnostic uses the v8 object-held-out train/validation split:

```text
x_t = state history + action history
h_t = future action channels 0:6
y_t = object action channels 6:15 + future contact state
```

`y_t` is never an input to the forward model. Forward and inverse models
use independent history encoders and are trained on the same data for 40
epochs with three seeds.

## Formal Result

The values below are three-seed means.

| Horizon | Inverse true MAE | Inverse shuffled MAE | Inverse constant MAE | Cycle MAE |
|---|---:|---:|---:|---:|
| H1 | 0.002521 | 0.002539 | 0.002949 | 0.002863 |
| H4 | 0.002944 | 0.002975 | 0.003324 | 0.003207 |
| H8 | 0.003684 | 0.003721 | 0.003642 | 0.003955 |

The inverse model improves over shuffled consequences by about 1.0-1.1%
at H4-H8. At H8 it is `1.14%` worse than the constant-action baseline.
This is insufficient action identifiability.

Forward action shuffling also leaves predictions almost unchanged:

| Horizon | Forward object translation | Action-shuffled object translation | Forward contact F1 | Action-shuffled contact F1 |
|---|---:|---:|---:|---:|
| H1 | 0.014357 | 0.014359 | 0.9471 | 0.9472 |
| H4 | 0.013108 | 0.013111 | 0.8442 | 0.8443 |
| H8 | 0.012963 | 0.012966 | 0.7099 | 0.7102 |

The forward model ignores the hand action. History shuffling changes contact
F1 by roughly 5-7 percentage points, so the model is using history to predict
the current interaction regime, not using the future hand action to predict
its consequence.

## Decision

The E2 identifiability gate fails. The current representation does not
support reliable consequence-to-action inversion, and the forward model does
not exhibit action sensitivity. Forward-inverse consistency must not be used
as a candidate reranker or refinement signal in this form.

This negative result is useful: ACID/WAV-style verification is not enough if
the underlying state/action representation does not expose an identifiable
causal effect. Before retrying, the project needs a richer action interface
or counterfactual response data, not another consistency loss.

## Evidence

Formal runs:

```text
/root/autodl-tmp/contact_action_20260914/forward_inverse_consistency_20260918/formal_v2
```

Aggregation script:

```text
scripts/summarize_forward_inverse_consistency.py
```
