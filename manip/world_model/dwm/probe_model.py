"""Passive probe-conditioned counterfactual ranking model."""

from __future__ import annotations

import torch
from torch import nn

from .config import ACTION_DIM
from .schema import STATE_DIM


class DWMProbeRankingModel(nn.Module):
    """Encode a probe history, then score candidate action chunks."""

    def __init__(
        self,
        hidden_size=128,
        response_dim=16,
        max_probe_horizon=4,
        max_candidate_horizon=8,
        oracle_context_dim=0,
    ):
        super().__init__()
        self.hidden_size = int(hidden_size)
        self.response_dim = int(response_dim)
        self.max_probe_horizon = int(max_probe_horizon)
        self.max_candidate_horizon = int(max_candidate_horizon)
        self.oracle_context_dim = int(oracle_context_dim)
        if min(
            self.hidden_size,
            self.response_dim,
            self.max_probe_horizon,
            self.max_candidate_horizon,
        ) < 1:
            raise ValueError("model dimensions must be positive")

        self.state_encoder = nn.Sequential(
            nn.Linear(STATE_DIM, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(ACTION_DIM, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.probe_rnn = nn.GRU(
            input_size=ACTION_DIM + 2 * STATE_DIM,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.response_head = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * response_dim),
        )
        self.oracle_context_encoder = None
        if self.oracle_context_dim > 0:
            self.oracle_context_encoder = nn.Sequential(
                nn.Linear(self.oracle_context_dim, hidden_size),
                nn.SiLU(),
                nn.Linear(hidden_size, 2 * response_dim),
            )

        self.candidate_rnn = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            batch_first=True,
        )
        self.target_encoder = nn.Sequential(
            nn.Linear(3, hidden_size),
            nn.SiLU(),
        )
        self.output_head = nn.Sequential(
            nn.Linear(hidden_size * 3 + response_dim, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, 10),
        )

    def _response_stats(
        self,
        initial_state,
        probe_action,
        probe_state,
        probe_mask,
        oracle_context,
    ):
        batch_size, probe_horizon, _ = probe_action.shape
        if probe_horizon != self.max_probe_horizon:
            raise ValueError(
                f"probe horizon must be {self.max_probe_horizon}"
            )
        state_feature = self.state_encoder(initial_state)
        previous_state = probe_state[:, :-1]
        next_state = probe_state[:, 1:]
        sequence = torch.cat([
            probe_action,
            next_state,
            next_state - previous_state,
        ], dim=-1)
        hidden, _ = self.probe_rnn(
            sequence,
            state_feature[None],
        )
        mask = probe_mask.to(dtype=hidden.dtype)
        denominator = mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        summary = (hidden * mask[:, :, None]).sum(dim=1) / denominator
        stats = self.response_head(torch.cat([
            state_feature,
            summary,
        ], dim=-1))
        if self.oracle_context_encoder is not None:
            if oracle_context is None:
                raise ValueError("oracle_context is required")
            stats = self.oracle_context_encoder(oracle_context)
        response_mu = stats[:, :self.response_dim]
        response_logvar = stats[:, self.response_dim:].clamp(-8.0, 5.0)
        return response_mu, response_logvar

    def forward(
        self,
        initial_state,
        probe_action,
        probe_state,
        probe_mask,
        post_probe_state,
        candidate_action,
        target_translation,
        oracle_context=None,
    ):
        if initial_state.shape != (initial_state.shape[0], STATE_DIM):
            raise ValueError("initial_state must be [B, 168]")
        if probe_action.ndim != 3 or probe_action.shape[2] != ACTION_DIM:
            raise ValueError("probe_action must be [B, L, 51]")
        if probe_state.ndim != 3 or probe_state.shape[2] != STATE_DIM:
            raise ValueError("probe_state must be [B, L+1, 168]")
        if probe_mask.shape != probe_action.shape[:2]:
            raise ValueError("probe_mask must match probe_action")
        if post_probe_state.shape != (post_probe_state.shape[0], STATE_DIM):
            raise ValueError("post_probe_state must be [B, 168]")
        if (
            candidate_action.ndim != 4
            or candidate_action.shape[2] != self.max_candidate_horizon
            or candidate_action.shape[3] != ACTION_DIM
        ):
            raise ValueError(
                "candidate_action must be [B, K, 8, 51]"
            )
        if target_translation.shape != (target_translation.shape[0], 3):
            raise ValueError("target_translation must be [B, 3]")

        response_mu, response_logvar = self._response_stats(
            initial_state,
            probe_action,
            probe_state,
            probe_mask,
            oracle_context,
        )
        batch_size, candidate_count = candidate_action.shape[:2]
        candidate_feature = self.action_encoder(
            candidate_action.reshape(
                batch_size * candidate_count,
                self.max_candidate_horizon,
                ACTION_DIM,
            )
        )
        candidate_feature, _ = self.candidate_rnn(candidate_feature)
        candidate_feature = candidate_feature[:, -1].reshape(
            batch_size,
            candidate_count,
            self.hidden_size,
        )
        post_feature = self.state_encoder(post_probe_state)
        response = torch.cat([
            post_feature,
            response_mu,
        ], dim=-1)
        response = response[:, None].expand(
            batch_size,
            candidate_count,
            response.shape[-1],
        )
        target_feature = self.target_encoder(target_translation)
        target_feature = target_feature[:, None].expand(
            batch_size,
            candidate_count,
            self.hidden_size,
        )
        output = self.output_head(torch.cat([
            response,
            candidate_feature,
            target_feature,
        ], dim=-1))
        return {
            "candidate_score": output[..., 0],
            "predicted_object_delta": output[..., 1:10],
            "response_mu": response_mu,
            "response_logvar": response_logvar,
        }
