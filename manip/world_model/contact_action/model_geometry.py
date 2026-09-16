"""Contact-action transition with an optional local geometry branch."""

from __future__ import annotations

import torch
from torch import nn

from .geometry import LocalGeometryEncoder, local_sdf_features
from .model import ContactActionTransition


class ContactActionGeometryTransition(ContactActionTransition):
    """Add a local SDF point encoder to the existing transition model."""

    def __init__(
        self,
        state_dim=34,
        action_dim=15,
        hidden_size=256,
        state_embedding=128,
        action_embedding=64,
        residual_scale=0.05,
        geometry_output_size=64,
        geometry_patch_grid=5,
        geometry_radius_normalized=0.08,
        geometry_mode="normal",
        residual_mask="palm",
    ):
        super().__init__(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_size=hidden_size,
            state_embedding=state_embedding,
            action_embedding=action_embedding,
            residual_scale=residual_scale,
            residual_mask=residual_mask,
        )
        if geometry_mode not in ("normal", "zero", "shuffle"):
            raise ValueError(f"unsupported geometry_mode: {geometry_mode}")
        self.state_embedding = state_embedding
        self.action_embedding = action_embedding
        self.geometry_mode = geometry_mode
        self.geometry_patch_grid = geometry_patch_grid
        self.geometry_radius_normalized = geometry_radius_normalized
        self.geometry_encoder = LocalGeometryEncoder(
            hidden_size=64,
            output_size=geometry_output_size,
        )
        self.geometry_project = nn.Sequential(
            nn.Linear(geometry_output_size, state_embedding),
            nn.SiLU(),
        )
        self.cell = nn.GRUCell(
            input_size=(
                state_embedding
                + action_embedding
                + 16
                + state_embedding
            ),
            hidden_size=hidden_size,
        )
        self._current_geometry_embedding = None
        self._collect_geometry_embeddings = False
        self.collected_geometry_embeddings = []

    def _geometry_features(
        self,
        state,
        object_indices,
        geometry_bank,
        geometry_mode,
    ):
        zero_features = torch.zeros(
            state.shape[0],
            2,
            self.geometry_patch_grid**3,
            5,
            device=state.device,
            dtype=state.dtype,
        )
        if geometry_mode == "zero":
            return zero_features
        if geometry_bank is None:
            raise ValueError("geometry_bank is required for this mode")
        geometry_state = state
        geometry_indices = object_indices
        if geometry_mode == "shuffle":
            batch_size = state.shape[0]
            if batch_size > 1:
                if self.training:
                    permutation = torch.randperm(
                        batch_size,
                        device=state.device,
                    )
                else:
                    permutation = torch.roll(
                        torch.arange(batch_size, device=state.device),
                        shifts=1,
                    )
                geometry_state = state.clone()
                geometry_state[..., 9:15] = state[
                    permutation,
                    9:15,
                ]
                geometry_indices = object_indices[permutation]
        return local_sdf_features(
            geometry_bank,
            geometry_state[..., 9:15].reshape(-1, 2, 3),
            geometry_indices,
            self.geometry_patch_grid,
            self.geometry_radius_normalized,
        )

    def _geometry_embedding(
        self,
        state,
        object_indices,
        geometry_bank,
        geometry_mode,
    ):
        hand_embeddings = self.encode_geometry_hands(
            state,
            object_indices,
            geometry_bank,
            geometry_mode,
        )
        return self.geometry_encoder.merge_hands(hand_embeddings)

    def encode_geometry_hands(
        self,
        state,
        object_indices,
        geometry_bank,
        geometry_mode=None,
    ):
        mode = geometry_mode or self.geometry_mode
        if mode == "zero":
            features = torch.zeros(
                state.shape[0],
                2,
                self.geometry_patch_grid**3,
                5,
                device=state.device,
                dtype=state.dtype,
            )
        else:
            features = self._geometry_features(
                state,
                object_indices,
                geometry_bank,
                mode,
            )
        return self.geometry_encoder.encode_hands(features)

    def step_hidden(self, hidden, state, action):
        normal = state[..., 26:32].reshape(*state.shape[:-1], 2, 3)
        palm_velocity = state[..., 15:21].reshape(*state.shape[:-1], 2, 3)
        object_velocity = state[..., 21:24].unsqueeze(-2)
        palm_action = action[..., 0:6].reshape(*action.shape[:-1], 2, 3)
        object_action = action[..., 6:9].unsqueeze(-2)
        velocity_normal = ((palm_velocity - object_velocity) * normal).sum(
            dim=-1
        )
        action_normal = ((palm_action - object_action) * normal).sum(dim=-1)
        interaction = torch.cat([velocity_normal, action_normal], dim=-1)
        if self._current_geometry_embedding is None:
            geometry_embedding = torch.zeros(
                state.shape[0],
                self.state_embedding,
                device=state.device,
                dtype=state.dtype,
            )
        else:
            geometry_embedding = self.geometry_project(
                self._current_geometry_embedding
            )
        inputs = torch.cat(
            [
                self.state_encoder(state),
                self.action_encoder(action),
                self.interaction_encoder(interaction),
                geometry_embedding,
            ],
            dim=-1,
        )
        return self.cell(inputs, hidden)

    def encode_history(
        self,
        state_history,
        action_history,
        geometry_bank=None,
        object_indices=None,
        geometry_mode=None,
    ):
        mode = geometry_mode or self.geometry_mode
        hidden = self.initial_hidden(
            state_history.shape[0],
            state_history.device,
            state_history.dtype,
        )
        for index in range(state_history.shape[1]):
            state = state_history[:, index]
            self._current_geometry_embedding = self._geometry_embedding(
                state,
                object_indices,
                geometry_bank,
                mode,
            )
            if self._collect_geometry_embeddings:
                self.collected_geometry_embeddings.append(
                    self._current_geometry_embedding
                )
            hidden = self.step_hidden(
                hidden,
                state,
                action_history[:, index],
            )
        return hidden

    def rollout(
        self,
        state_history,
        action_history,
        future_actions,
        teacher_states=None,
        teacher_forcing_ratio=1.0,
        geometry_bank=None,
        object_indices=None,
        geometry_mode=None,
        use_learned_residual=True,
    ):
        if future_actions.ndim != 3:
            raise ValueError("future_actions must be [B, H, A]")
        if teacher_states is not None and teacher_states.shape != (
            future_actions.shape[0],
            future_actions.shape[1],
            self.state_dim,
        ):
            raise ValueError("teacher_states shape does not match future actions")
        if not 0.0 <= teacher_forcing_ratio <= 1.0:
            raise ValueError("teacher_forcing_ratio must lie in [0, 1]")
        mode = geometry_mode or self.geometry_mode
        self.collected_geometry_embeddings = []
        self._collect_geometry_embeddings = True
        hidden = self.encode_history(
            state_history,
            action_history,
            geometry_bank=geometry_bank,
            object_indices=object_indices,
            geometry_mode=mode,
        )
        input_state = state_history[:, -1]
        predicted_states = []
        contact_probabilities = []
        contact_logits = []
        onset_logits = []
        release_logits = []
        residuals = []

        for horizon_index in range(future_actions.shape[1]):
            self._current_geometry_embedding = self._geometry_embedding(
                input_state,
                object_indices,
                geometry_bank,
                mode,
            )
            self.collected_geometry_embeddings.append(
                self._current_geometry_embedding
            )
            step_output = self.step(
                hidden,
                input_state,
                future_actions[:, horizon_index],
                use_learned_residual=use_learned_residual,
            )
            hidden = step_output["hidden"]
            next_state = step_output["state"]
            predicted_states.append(next_state)
            contact_probabilities.append(
                step_output["contact_probability"]
            )
            contact_logits.append(step_output["contact_logits"])
            onset_logits.append(step_output["onset_logits"])
            release_logits.append(step_output["release_logits"])
            residuals.append(step_output["residual"])
            if teacher_states is None or teacher_forcing_ratio == 0.0:
                input_state = next_state
            elif teacher_forcing_ratio == 1.0:
                input_state = teacher_states[:, horizon_index]
            else:
                use_teacher = (
                    torch.rand(
                        state_history.shape[0],
                        1,
                        device=state_history.device,
                    )
                    < teacher_forcing_ratio
                )
                input_state = torch.where(
                    use_teacher,
                    teacher_states[:, horizon_index],
                    next_state,
                )

        self._collect_geometry_embeddings = False
        return {
            "states": torch.stack(predicted_states, dim=1),
            "contact_probability": torch.stack(
                contact_probabilities,
                dim=1,
            ),
            "contact_logits": torch.stack(contact_logits, dim=1),
            "onset_logits": torch.stack(onset_logits, dim=1),
            "release_logits": torch.stack(release_logits, dim=1),
            "geometry_embeddings": torch.stack(
                self.collected_geometry_embeddings,
                dim=1,
            ),
            "residuals": torch.stack(residuals, dim=1),
        }
