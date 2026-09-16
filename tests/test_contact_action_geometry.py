from __future__ import annotations

import torch

from manip.world_model.contact_action.geometry import (
    LocalGeometryBank,
    LocalGeometryEncoder,
    local_sdf_features,
)
from manip.world_model.contact_action.model_geometry import (
    ContactActionGeometryTransition,
)


def make_bank(device="cpu"):
    return LocalGeometryBank(
        names=("object",),
        grids=torch.zeros(1, 1, 4, 4, 4, device=device),
        centroids=torch.zeros(1, 3, device=device),
        extents=torch.ones(1, 3, device=device),
        scales=torch.ones(1, device=device),
    )


def test_local_sdf_features_shape():
    bank = make_bank()
    palm = torch.zeros(2, 2, 3)
    object_indices = torch.zeros(2, dtype=torch.long)
    features = local_sdf_features(
        bank,
        palm,
        object_indices,
        patch_grid=3,
        radius_normalized=0.08,
    )
    assert features.shape == (2, 2, 27, 5)
    assert torch.isfinite(features).all()
    assert features[..., 4].all()


def test_geometry_encoder_shape():
    encoder = LocalGeometryEncoder(hidden_size=8, output_size=4)
    output = encoder(torch.zeros(3, 2, 27, 5))
    assert output.shape == (3, 4)


def test_geometry_transition_rollout_modes():
    bank = make_bank()
    for mode in ("normal", "zero", "shuffle"):
        model = ContactActionGeometryTransition(
            hidden_size=16,
            residual_scale=0.0,
            geometry_output_size=8,
            geometry_patch_grid=3,
            geometry_radius_normalized=0.08,
            geometry_mode=mode,
        )
        state_history = torch.zeros(2, 4, 34)
        action_history = torch.zeros(2, 4, 15)
        future_actions = torch.zeros(2, 3, 15)
        output = model.rollout(
            state_history,
            action_history,
            future_actions,
            teacher_states=None,
            teacher_forcing_ratio=0.0,
            geometry_bank=bank,
            object_indices=torch.zeros(2, dtype=torch.long),
            geometry_mode=mode,
        )
        assert output["states"].shape == (2, 3, 34)
        assert output["contact_logits"].shape == (2, 3, 2)
