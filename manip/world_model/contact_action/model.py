"""Small action-conditioned contact/response transition model."""

from __future__ import annotations

import torch
from torch import nn

from .features import ACTION_DIM, CONTACT_SLICE, NONCONTACT_DIM, STATE_DIM


def residual_mask_from_mode(mode):
    """Return the non-contact state dimensions allowed to receive residual."""
    mask = torch.zeros(NONCONTACT_DIM)
    if mode == "none":
        return mask
    if mode == "palm":
        mask[9:21] = 1.0
        mask[24:32] = 1.0
        return mask
    if mode == "full":
        mask[:] = 1.0
        return mask
    raise ValueError(f"unsupported residual mask mode: {mode}")


def _rotation_6d_to_matrix(rotation_6d):
    """Decode first-two-rows 6D rotations into matrices."""
    row1 = rotation_6d[..., 0:3]
    row1 = row1 / row1.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    row2 = rotation_6d[..., 3:6]
    row2 = (
        row2 - (row2 * row1).sum(dim=-1, keepdim=True) * row1
    )
    row2 = row2 / row2.norm(dim=-1, keepdim=True).clamp_min(1e-6)
    row3 = torch.cross(row1, row2, dim=-1)
    return torch.stack([row1, row2, row3], dim=-2)


def _matrix_to_rotation_6d(rotation):
    """Encode matrices with the first-two-rows convention."""
    return rotation[..., :2, :].reshape(*rotation.shape[:-2], 6)


class ContactActionTransition(nn.Module):
    """Recurrent transition ``s_{t+1} = F(s_t, a_t)``.

    The final two state dimensions are contact probabilities. They are
    decoded through a separate sigmoid head so rollout feedback uses the
    model's own contact estimate rather than GT labels.
    """

    def __init__(
        self,
        state_dim=STATE_DIM,
        action_dim=ACTION_DIM,
        hidden_size=256,
        state_embedding=128,
        action_embedding=64,
        residual_scale=0.05,
        residual_mask="palm",
    ):
        super().__init__()
        if state_dim != STATE_DIM:
            raise ValueError(f"state_dim must be {STATE_DIM}")
        if action_dim != ACTION_DIM:
            raise ValueError(f"action_dim must be {ACTION_DIM}")

        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_size = hidden_size
        self.residual_scale = float(residual_scale)
        if self.residual_scale < 0:
            raise ValueError("residual_scale must be non-negative")
        self.residual_mask_mode = residual_mask
        self.register_buffer(
            "residual_mask",
            residual_mask_from_mode(residual_mask),
            persistent=False,
        )

        self.state_encoder = nn.Sequential(
            nn.Linear(state_dim, state_embedding),
            nn.SiLU(),
            nn.Linear(state_embedding, state_embedding),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, action_embedding),
            nn.SiLU(),
            nn.Linear(action_embedding, action_embedding),
        )
        self.interaction_encoder = nn.Sequential(
            nn.Linear(4, 16),
            nn.SiLU(),
        )
        self.cell = nn.GRUCell(
            input_size=state_embedding + action_embedding + 16,
            hidden_size=hidden_size,
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, NONCONTACT_DIM + 4),
        )
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def initial_hidden(self, batch_size, device, dtype):
        return torch.zeros(batch_size, self.hidden_size, device=device, dtype=dtype)

    def encode_history(self, state_history, action_history):
        if state_history.shape[:2] != action_history.shape[:2]:
            raise ValueError("state and action histories must share [B, L]")
        hidden = self.initial_hidden(
            state_history.shape[0],
            state_history.device,
            state_history.dtype,
        )
        for index in range(state_history.shape[1]):
            hidden = self.step_hidden(
                hidden,
                state_history[:, index],
                action_history[:, index],
            )
        return hidden

    def step_hidden(self, hidden, state, action):
        normal = state[..., 26:32].reshape(*state.shape[:-1], 2, 3)
        palm_velocity = state[..., 15:21].reshape(*state.shape[:-1], 2, 3)
        object_velocity = state[..., 21:24].unsqueeze(-2)
        palm_action = action[..., 0:6].reshape(*action.shape[:-1], 2, 3)
        object_action = action[..., 6:9].unsqueeze(-2)
        velocity_normal = (
            (palm_velocity - object_velocity) * normal
        ).sum(dim=-1)
        action_normal = (
            (palm_action - object_action) * normal
        ).sum(dim=-1)
        interaction = torch.cat([velocity_normal, action_normal], dim=-1)
        inputs = torch.cat(
            [
                self.state_encoder(state),
                self.action_encoder(action),
                self.interaction_encoder(interaction),
            ],
            dim=-1,
        )
        return self.cell(inputs, hidden)

    def step(
        self,
        hidden,
        state,
        action,
        use_learned_residual=True,
    ):
        hidden = self.step_hidden(hidden, state, action)
        raw = self.head(hidden)
        residual = raw[..., :NONCONTACT_DIM]
        event_logits = raw[..., NONCONTACT_DIM:]
        onset_logits = event_logits[..., 0:2]
        release_logits = event_logits[..., 2:4]
        current_object_rotation = _rotation_6d_to_matrix(state[..., 3:9])
        action_object_rotation = _rotation_6d_to_matrix(action[..., 9:15])
        next_object_rotation = (
            action_object_rotation @ current_object_rotation
        )
        object_translation_delta = (
            action[..., 6:9].unsqueeze(-2) @ current_object_rotation
        ).squeeze(-2)
        action_groups = action.reshape(*action.shape[:-1], 5, 3)
        action_in_next_object_frame = (
            action_groups @ action_object_rotation.transpose(-1, -2)
        ).reshape(*action.shape)
        normal = state[..., 26:32].reshape(*state.shape[:-1], 2, 3)
        palm_action = action[..., 0:6].reshape(*action.shape[:-1], 2, 3)
        object_action = action[..., 6:9].unsqueeze(-2)
        clearance_delta = (
            (palm_action - object_action) * normal
        ).sum(dim=-1)
        action_base = torch.cat(
            [
                object_translation_delta,
                _matrix_to_rotation_6d(next_object_rotation),
                action_in_next_object_frame[..., 0:6],  # palm position
                action_in_next_object_frame[..., 0:6],  # palm velocity
                action_in_next_object_frame[..., 6:9],  # object velocity
                clearance_delta,
                torch.zeros_like(
                    state[..., 26:NONCONTACT_DIM]
                ),  # normal persistence
            ],
            dim=-1,
        )
        bounded_residual = torch.tanh(
            residual * self.residual_mask
        )
        applied_residual = (
            self.residual_scale * bounded_residual
            if use_learned_residual
            else torch.zeros_like(bounded_residual)
        )
        noncontact = (
            state[..., :NONCONTACT_DIM]
            + action_base
            + applied_residual
        )
        current_contact = state[..., CONTACT_SLICE]
        current_contact = current_contact.clamp(0.0, 1.0)
        contact_probability = (
            current_contact
            * (1.0 - torch.sigmoid(release_logits))
            + (1.0 - current_contact) * torch.sigmoid(onset_logits)
        )
        contact_probability = contact_probability.clamp(
            1e-6,
            1.0 - 1e-6,
        )
        contact_logits = torch.logit(contact_probability)
        contact_probability = torch.sigmoid(contact_logits)
        next_state = torch.cat([noncontact, contact_probability], dim=-1)
        return {
            "hidden": hidden,
            "state": next_state,
            "contact_probability": contact_probability,
            "contact_logits": contact_logits,
            "onset_logits": onset_logits,
            "release_logits": release_logits,
            "residual": bounded_residual,
            "applied_residual": applied_residual,
        }

    def rollout(
        self,
        state_history,
        action_history,
        future_actions,
        teacher_states=None,
        teacher_forcing_ratio=1.0,
        use_learned_residual=True,
    ):
        """Run a short rollout.

        teacher_forcing_ratio=1 uses GT future state as the next input.
        Values below 1 randomly feed predicted states, enabling mixed or
        free-running rollout training.
        """
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

        hidden = self.encode_history(state_history, action_history)
        input_state = state_history[:, -1]
        predicted_states = []
        contact_probabilities = []
        contact_logits = []
        onset_logits = []
        release_logits = []
        residuals = []

        for horizon_index in range(future_actions.shape[1]):
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

            if teacher_states is None:
                input_state = next_state
                continue

            if teacher_forcing_ratio == 1.0:
                input_state = teacher_states[:, horizon_index]
            elif teacher_forcing_ratio == 0.0:
                input_state = next_state
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

        return {
            "states": torch.stack(predicted_states, dim=1),
            "contact_probability": torch.stack(
                contact_probabilities,
                dim=1,
            ),
            "contact_logits": torch.stack(contact_logits, dim=1),
            "onset_logits": torch.stack(onset_logits, dim=1),
            "release_logits": torch.stack(release_logits, dim=1),
            "residuals": torch.stack(residuals, dim=1),
        }
