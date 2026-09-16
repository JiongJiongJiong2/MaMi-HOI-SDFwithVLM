"""Checkpoint compatibility helpers for controlled fine-tuning."""

from typing import Dict, Iterable, Mapping, Tuple

import torch


FINETUNE_SDF_ONLY_MODEL_KEYS = (
    "denoise_fn.gaa_block.sdf_bias_mlp.0.weight",
    "denoise_fn.gaa_block.sdf_bias_mlp.0.bias",
    "denoise_fn.gaa_block.sdf_bias_mlp.2.weight",
    "denoise_fn.gaa_block.sdf_bias_mlp.2.bias",
    "denoise_fn.gaa_block.sdf_feat_proj.weight",
    "denoise_fn.gaa_block.sdf_feat_proj.bias",
    "denoise_fn.sdf_value_proj.0.weight",
    "denoise_fn.sdf_value_proj.0.bias",
    "denoise_fn.sdf_value_proj.2.weight",
    "denoise_fn.sdf_value_proj.2.bias",
)


def derive_ema_allowed_missing_keys(
    current_ema_state_dict: Mapping[str, torch.Tensor],
    allowed_model_missing_keys: Iterable[str],
) -> Tuple[str, ...]:
    """Map allowed model keys to the EMA wrappers that are actually present."""
    current_keys = set(current_ema_state_dict)
    allowed = []
    for prefix in ("online_model.", "ema_model."):
        for key in allowed_model_missing_keys:
            candidate = prefix + key
            if candidate in current_keys:
                allowed.append(candidate)
    return tuple(allowed)


def build_compatible_state_dict(
    current_state_dict: Mapping[str, torch.Tensor],
    checkpoint_state_dict: Mapping[str, torch.Tensor],
    allowed_missing_keys: Iterable[str] = (),
    *,
    context: str = "state_dict",
) -> Tuple[Dict[str, torch.Tensor], Dict[str, object]]:
    """Fill only explicitly allowed missing keys, then require an exact match."""
    current_keys = set(current_state_dict)
    checkpoint_keys = set(checkpoint_state_dict)
    allowed_missing = set(allowed_missing_keys)

    unknown_allowed = sorted(allowed_missing - current_keys)
    if unknown_allowed:
        raise ValueError(
            f"{context}: allowed missing keys are not present in the current "
            f"model: {unknown_allowed}"
        )

    missing = sorted(current_keys - checkpoint_keys)
    disallowed_missing = sorted(set(missing) - allowed_missing)
    unexpected = sorted(checkpoint_keys - current_keys)

    mismatches = []
    for key in sorted(current_keys & checkpoint_keys):
        current_value = current_state_dict[key]
        checkpoint_value = checkpoint_state_dict[key]
        current_shape = tuple(current_value.shape)
        checkpoint_shape = tuple(checkpoint_value.shape)
        current_dtype = str(current_value.dtype)
        checkpoint_dtype = str(checkpoint_value.dtype)
        if (
            current_shape != checkpoint_shape
            or current_dtype != checkpoint_dtype
        ):
            mismatches.append(
                {
                    "key": key,
                    "current_shape": list(current_shape),
                    "checkpoint_shape": list(checkpoint_shape),
                    "current_dtype": current_dtype,
                    "checkpoint_dtype": checkpoint_dtype,
                }
            )

    problems = []
    if disallowed_missing:
        problems.append(f"missing keys: {disallowed_missing}")
    if unexpected:
        problems.append(f"unexpected keys: {unexpected}")
    if mismatches:
        problems.append(f"shape/dtype mismatches: {mismatches}")
    if problems:
        raise RuntimeError(f"{context}: incompatible checkpoint; " + "; ".join(problems))

    compatible = dict(checkpoint_state_dict)
    for key in missing:
        compatible[key] = current_state_dict[key].detach().clone()

    report = {
        "context": context,
        "current_key_count": len(current_keys),
        "checkpoint_key_count": len(checkpoint_keys),
        "allowed_missing_keys": sorted(allowed_missing),
        "initialized_missing_keys": missing,
        "unexpected_keys": unexpected,
        "shape_or_dtype_mismatch_count": len(mismatches),
    }
    return compatible, report
