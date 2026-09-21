"""Compact action-conditioned object-response transition model."""

from __future__ import annotations

import torch
from torch import nn

from .config import ACTION_DIM
from .schema import CONTACT_MODE_LABELS, STATE_DIM


OUTPUT_DIM = 9 + len(CONTACT_MODE_LABELS) + 3


class DWMTransitionModel(nn.Module):
    def __init__(
        self,
        hidden_size=192,
        action_embedding=128,
        max_horizon=8,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.action_embedding = int(action_embedding)
        self.max_horizon = int(max_horizon)
        if self.hidden_size < 1 or self.action_embedding < 1:
            raise ValueError("hidden sizes must be positive")
        if self.max_horizon < 1:
            raise ValueError("max_horizon must be positive")

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM, action_embedding),
            nn.SiLU(),
            nn.Linear(action_embedding, action_embedding),
            nn.SiLU(),
        )
        self.action_gru = nn.GRU(
            input_size=action_embedding,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.horizon_embedding = nn.Parameter(
            torch.randn(1, max_horizon, hidden_size) * 0.02
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, OUTPUT_DIM),
        )

    def forward(self, initial_state, action_chunk):
        if initial_state.ndim != 2 or initial_state.shape[1] != STATE_DIM:
            raise ValueError(
                f"initial_state must be [B, {STATE_DIM}]"
            )
        if (
            action_chunk.ndim != 3
            or action_chunk.shape[1] != self.max_horizon
            or action_chunk.shape[2] != ACTION_DIM
        ):
            raise ValueError(
                "action_chunk must be "
                f"[B, {self.max_horizon}, {ACTION_DIM}]"
            )
        state_feature = self.state_encoder(initial_state)
        action_feature = self.action_encoder(action_chunk)
        action_feature, _ = self.action_gru(action_feature)
        action_feature = action_feature + self.horizon_embedding
        hidden = torch.cat(
            [
                state_feature[:, None].expand_as(action_feature),
                action_feature,
            ],
            dim=-1,
        )
        output = self.head(hidden)
        return {
            "object_delta": output[..., :9],
            "contact_mode_logits": output[..., 9:13],
            "contact_impulse_normalized": output[..., 13:16],
        }
