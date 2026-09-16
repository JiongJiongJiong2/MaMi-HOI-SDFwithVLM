"""Loading and reduction helpers for the frozen HOI-Dyn module."""

from pathlib import Path

import torch

from .model import Dynamics
from .utils import read_yaml


def load_hoidyn_dynamics(config_path, checkpoint_path, device):
    """Load a frozen HOI-Dyn model from its official checkpoint format."""
    config_path = Path(config_path)
    checkpoint_path = Path(checkpoint_path)
    if not config_path.is_file():
        raise FileNotFoundError(config_path)
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)

    config = read_yaml(str(config_path))
    if "model" not in config:
        raise KeyError("HOI-Dyn config must contain a 'model' section")

    model = Dynamics(config["model"])
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("model_state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.to(device)
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    return model


def reduce_hoidyn_regulate_loss(
    regulate_loss,
    loss_type="pc",
    rot_weight=0.05,
):
    """Reduce HOI-Dyn losses to one scalar per batch item."""
    if loss_type == "pc":
        keys = ("pc_loss",)
    elif loss_type == "trans_rot":
        keys = ("trans_loss", "rot_loss")
    elif loss_type == "trans":
        keys = ("trans_loss",)
    else:
        raise ValueError(f"Unsupported HOI-Dyn loss_type: {loss_type}")

    per_sample = None
    stats = {}
    for key in keys:
        if key not in regulate_loss:
            raise KeyError(f"Missing HOI-Dyn regulate loss: {key}")
        value = regulate_loss[key]
        if value.ndim == 0:
            value = value.reshape(1)
        reduced = value.reshape(value.shape[0], -1).mean(dim=1)
        weight = rot_weight if key == "rot_loss" else 1.0
        per_sample = reduced * weight if per_sample is None else per_sample + reduced * weight
        stats[key] = float(value.detach().mean().item())

    stats["total"] = float(per_sample.detach().mean().item())
    stats["loss_type"] = loss_type
    stats["rot_weight"] = float(rot_weight)
    return per_sample, stats
