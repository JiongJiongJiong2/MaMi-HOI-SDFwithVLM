"""Forward and inverse models for the E2 consistency diagnostic."""

from __future__ import annotations

import torch
from torch import nn


STATE_DIM = 34
ACTION_DIM = 15
HAND_ACTION_DIM = 6
OBJECT_CONSEQUENCE_DIM = 9
CONTACT_DIM = 2
CONSEQUENCE_DIM = OBJECT_CONSEQUENCE_DIM + CONTACT_DIM


class HistoryEncoder(nn.Module):
    def __init__(
        self,
        history=4,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        output_size=256,
    ):
        super().__init__()
        input_size = history * (state_dim + action_dim)
        self.network = nn.Sequential(
            nn.Linear(input_size, 512),
            nn.LayerNorm(512),
            nn.SiLU(),
            nn.Linear(512, output_size),
            nn.SiLU(),
        )

    def forward(self, state_history, action_history):
        features = torch.cat(
            [
                state_history.reshape(state_history.shape[0], -1),
                action_history.reshape(action_history.shape[0], -1),
            ],
            dim=-1,
        )
        return self.network(features)


def _horizon_features(reference, horizon, max_horizon):
    if horizon < 1 or horizon > max_horizon:
        raise ValueError("horizon is outside the configured range")
    one_hot = torch.zeros(
        reference.shape[0],
        max_horizon,
        device=reference.device,
        dtype=reference.dtype,
    )
    one_hot[:, horizon - 1] = 1.0
    return one_hot


class ForwardConsequenceModel(nn.Module):
    def __init__(
        self,
        history=4,
        hidden_size=256,
        max_horizon=8,
    ):
        super().__init__()
        self.max_horizon = max_horizon
        self.encoder = HistoryEncoder(
            history=history,
            output_size=hidden_size,
        )
        self.head = nn.Sequential(
            nn.Linear(
                hidden_size + HAND_ACTION_DIM + max_horizon,
                256,
            ),
            nn.SiLU(),
            nn.Linear(256, CONSEQUENCE_DIM),
        )

    def forward(
        self,
        state_history,
        action_history,
        future_hand_actions,
    ):
        context = self.encoder(state_history, action_history)
        outputs = []
        for horizon in range(1, self.max_horizon + 1):
            action = future_hand_actions[:, horizon - 1]
            inputs = torch.cat(
                [
                    context,
                    action,
                    _horizon_features(
                        action,
                        horizon,
                        self.max_horizon,
                    ),
                ],
                dim=-1,
            )
            outputs.append(self.head(inputs))
        return torch.stack(outputs, dim=1)


class InverseActionModel(nn.Module):
    def __init__(
        self,
        history=4,
        hidden_size=256,
        max_horizon=8,
    ):
        super().__init__()
        self.max_horizon = max_horizon
        self.encoder = HistoryEncoder(
            history=history,
            output_size=hidden_size,
        )
        self.head = nn.Sequential(
            nn.Linear(
                hidden_size + CONSEQUENCE_DIM + max_horizon,
                256,
            ),
            nn.SiLU(),
            nn.Linear(256, HAND_ACTION_DIM),
        )

    def forward(
        self,
        state_history,
        action_history,
        consequences,
    ):
        context = self.encoder(state_history, action_history)
        outputs = []
        for horizon in range(1, self.max_horizon + 1):
            consequence = consequences[:, horizon - 1]
            inputs = torch.cat(
                [
                    context,
                    consequence,
                    _horizon_features(
                        consequence,
                        horizon,
                        self.max_horizon,
                    ),
                ],
                dim=-1,
            )
            outputs.append(self.head(inputs))
        return torch.stack(outputs, dim=1)


def build_history_context_inputs(split):
    return split["state_history"], split["action_history"]


def build_hand_actions(future_actions):
    return future_actions[..., :HAND_ACTION_DIM]


def build_consequences(future_actions, future_states):
    return torch.cat(
        [
            future_actions[..., HAND_ACTION_DIM:ACTION_DIM],
            future_states[..., 32:34],
        ],
        dim=-1,
    )


def decode_forward_consequences(forward_output):
    object_delta = forward_output[..., :OBJECT_CONSEQUENCE_DIM]
    contact_probability = torch.sigmoid(
        forward_output[..., OBJECT_CONSEQUENCE_DIM:]
    )
    return torch.cat([object_delta, contact_probability], dim=-1)

