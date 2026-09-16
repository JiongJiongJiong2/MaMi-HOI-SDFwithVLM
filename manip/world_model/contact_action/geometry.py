"""Local object-SDF geometry features for the contact-action model."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import nn

from manip.model.sdf_utils import sample_object_sdf_at_points
from scripts.build_contact_action_dataset import (
    load_object_scale,
    load_object_sdf,
)


GEOMETRY_FEATURE_DIM = 5


@dataclass(frozen=True)
class LocalGeometryBank:
    names: tuple[str, ...]
    grids: torch.Tensor
    centroids: torch.Tensor
    extents: torch.Tensor
    scales: torch.Tensor

    def to(self, device):
        return LocalGeometryBank(
            names=self.names,
            grids=self.grids.to(device),
            centroids=self.centroids.to(device),
            extents=self.extents.to(device),
            scales=self.scales.to(device),
        )


class LocalGeometryEncoder(nn.Module):
    """Shared left/right point encoder for local SDF volumes."""

    def __init__(
        self,
        input_size=GEOMETRY_FEATURE_DIM,
        hidden_size=64,
        output_size=64,
    ):
        super().__init__()
        self.point_encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.SiLU(),
        )
        self.hand_encoder = nn.Sequential(
            nn.Linear(2 * hidden_size, hidden_size),
            nn.SiLU(),
            nn.Linear(hidden_size, output_size),
        )

    def encode_hands(self, features):
        if features.ndim != 4:
            raise ValueError(
                f"geometry features must be [B, 2, K, C], got {features.shape}"
            )
        batch_size, hand_count, point_count, feature_dim = features.shape
        if hand_count != 2 or feature_dim != GEOMETRY_FEATURE_DIM:
            raise ValueError(
                "geometry features must have two hands and five channels"
            )
        encoded = self.point_encoder(
            features.reshape(batch_size * hand_count, point_count, feature_dim)
        )
        return encoded.max(dim=1).values.reshape(
            batch_size,
            hand_count,
            -1,
        )

    def merge_hands(self, hand_embeddings):
        if hand_embeddings.ndim != 3 or hand_embeddings.shape[1] != 2:
            raise ValueError(
                "hand_embeddings must be [B, 2, D], "
                f"got {hand_embeddings.shape}"
            )
        return self.hand_encoder(
            hand_embeddings.reshape(hand_embeddings.shape[0], -1)
        )

    def forward(self, features):
        return self.merge_hands(self.encode_hands(features))


def build_geometry_bank(data_root, object_names, device):
    """Load canonical SDF grids for all requested objects."""

    scale_cache = {}
    sdf_cache = {}
    grids = []
    centroids = []
    extents = []
    scales = []
    for object_name in object_names:
        grid, centroid, extents_value = load_object_sdf(
            Path(data_root),
            object_name,
            sdf_cache,
        )
        grids.append(grid)
        centroids.append(centroid)
        extents.append(extents_value)
        scales.append(
            torch.tensor(
                [
                    load_object_scale(
                        Path(data_root),
                        object_name,
                        scale_cache,
                    )
                ],
                dtype=torch.float32,
            )
        )
    return LocalGeometryBank(
        names=tuple(object_names),
        grids=torch.cat(grids, dim=0).to(device),
        centroids=torch.cat(centroids, dim=0).to(device),
        extents=torch.cat(extents, dim=0).to(device),
        scales=torch.cat(scales, dim=0).to(device),
    )


def local_sdf_features(
    bank,
    palm_normalized,
    object_indices,
    patch_grid,
    radius_normalized,
):
    """Create a fixed local signed-distance volume around each palm."""

    if palm_normalized.ndim != 3 or palm_normalized.shape[1:] != (2, 3):
        raise ValueError(
            f"palm_normalized must be [B, 2, 3], got {palm_normalized.shape}"
        )
    if object_indices.shape != (palm_normalized.shape[0],):
        raise ValueError("object_indices must have shape [B]")
    if patch_grid < 2:
        raise ValueError("patch_grid must be at least 2")
    if not 0 < radius_normalized <= 0.5:
        raise ValueError("radius_normalized must lie in (0, 0.5]")

    batch_size = palm_normalized.shape[0]
    coordinates = torch.linspace(
        -radius_normalized,
        radius_normalized,
        patch_grid,
        device=palm_normalized.device,
        dtype=palm_normalized.dtype,
    )
    offsets = torch.stack(
        torch.meshgrid(coordinates, coordinates, coordinates, indexing="ij"),
        dim=-1,
    ).reshape(-1, 3)
    point_count = offsets.shape[0]
    offsets = offsets[None, None].expand(
        batch_size,
        2,
        point_count,
        3,
    )
    points = palm_normalized[:, :, None, :] + offsets
    features = torch.zeros(
        batch_size,
        2,
        point_count,
        GEOMETRY_FEATURE_DIM,
        device=palm_normalized.device,
        dtype=palm_normalized.dtype,
    )

    for object_index in torch.unique(object_indices):
        object_mask = object_indices == object_index
        object_index_int = int(object_index.item())
        scale = bank.scales[object_index_int]
        points_object = points[object_mask].reshape(-1, point_count, 3) * scale
        point_batch = points_object.shape[0]
        grid = bank.grids[object_index_int : object_index_int + 1]
        grid = grid.expand(point_batch, -1, -1, -1, -1)
        centroid = bank.centroids[
            object_index_int : object_index_int + 1
        ].expand(point_batch, -1)
        extents = bank.extents[
            object_index_int : object_index_int + 1
        ].expand(point_batch, -1)
        signed_distance, valid = sample_object_sdf_at_points(
            grid,
            points_object,
            centroid,
            extents,
            return_valid_mask=True,
        )
        signed_distance = signed_distance.reshape(
            int(object_mask.sum().item()),
            2,
            point_count,
        ) / scale
        valid = valid.reshape(
            int(object_mask.sum().item()),
            2,
            point_count,
        )
        signed_distance = torch.where(
            valid,
            signed_distance,
            torch.zeros_like(signed_distance),
        )
        selected = features[object_mask]
        selected[..., 0:3] = offsets[object_mask]
        selected[..., 3] = signed_distance
        selected[..., 4] = valid.to(features.dtype)
        features[object_mask] = selected
    return features
