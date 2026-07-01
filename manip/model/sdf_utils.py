"""
SDF (Signed Distance Field) utility functions for MaMi-HOI.

Provides batched, differentiable SDF queries on precomputed local SDF volumes,
gradient computation via finite differences, and hand contact point extraction.

All functions are pure (no nn.Module) and designed to work with
F.grid_sample on 3D volumetric SDF grids.
"""

import torch
import torch.nn.functional as F


def sample_sdf_at_points(sdf_grid, query_points, origin, voxel_size):
    """Query SDF values at arbitrary 3D points using F.grid_sample.

    This is the batched, differentiable version of the evaluation-time
    compute_signed_distances() in the trainer. It operates on local SDF
    volumes (e.g., 64^3) instead of full 256^3 grids.

    Args:
        sdf_grid: [B, 1, D, H, W] local SDF volume (e.g., [B, 1, 64, 64, 64]).
                  SDF values are stored in *normalized* coordinates (i.e., the
                  grid stores distance / (voxel_size / 2)).
        query_points: [B, N, 3] world-coordinate query points.
        origin: [B, 3] minimum corner of the local bounding box.
        voxel_size: scalar or [B, 1] full extent (max dimension) of the
                    bounding box. Used for normalization to [-1, 1].

    Returns:
        sdf_values: [B, N, 1] signed distance values in world units.
    """
    # Normalize query points to [-1, 1] range for grid_sample
    # Following the same convention as compute_signed_distances():
    #   query_norm = (query - centroid) * 2 / extents.max()
    # Here origin is the min corner, so centroid = origin + voxel_size/2
    # But the existing code uses (query - centroid) * 2 / max_extent,
    # which maps [centroid - max_extent/2, centroid + max_extent/2] -> [-1, 1]
    # Since origin = centroid - max_extent/2, this is equivalent to:
    #   query_norm = (query - origin) * 2 / voxel_size - 1
    # However, the existing code uses (query - centroid) * 2 / max_extent
    # where centroid is the center and max_extent is the full size.
    # Let's follow the same pattern: origin here is the center (centroid),
    # and voxel_size is the max extent.
    if isinstance(voxel_size, (int, float)):
        query_norm = (query_points - origin[:, None, :]) * 2.0 / voxel_size
    else:
        # voxel_size: [B, 1] or [B]
        if voxel_size.dim() == 1:
            voxel_size = voxel_size[:, None]  # [B, 1]
        query_norm = (query_points - origin[:, None, :]) * 2.0 / voxel_size[:, :, None]

    # Swap axis order to (depth, height, width) as required by F.grid_sample
    # This matches the existing compute_signed_distances() convention
    query_norm = query_norm[..., [2, 1, 0]]

    # Reshape for F.grid_sample: grid should be [B, D_out, H_out, W_out, 3]
    # We want to query N points per batch, so D_out=1, H_out=1, W_out=N
    query_norm = query_norm[:, None, None, :, :]  # [B, 1, 1, N, 3]

    # F.grid_sample: input [B, C, D_in, H_in, W_in], grid [B, D_out, H_out, W_out, 3]
    # Output: [B, C, D_out, H_out, W_out]
    signed_dists = F.grid_sample(
        sdf_grid, query_norm,
        padding_mode='border',
        align_corners=True
    )  # [B, 1, 1, 1, N]

    # Reshape: squeeze spatial dims and permute
    signed_dists = signed_dists.squeeze(2).squeeze(2)  # [B, 1, N]
    signed_dists = signed_dists.permute(0, 2, 1)  # [B, N, 1]

    # Scale back from normalized coordinates to world units
    # The SDF grid stores distances in normalized space, so we multiply by
    # voxel_size / 2 to convert to world units (same as compute_signed_distances)
    if isinstance(voxel_size, (int, float)):
        signed_dists = signed_dists * (voxel_size / 2.0)
    else:
        if voxel_size.dim() == 1:
            voxel_size = voxel_size[:, None]  # [B, 1]
        signed_dists = signed_dists * (voxel_size[:, :, None] / 2.0)

    return signed_dists


def compute_sdf_gradients_fd(sdf_grid, query_points, origin, voxel_size, eps=1e-3):
    """Compute SDF gradients via central finite differences.

    For each axis, queries SDF at point +/- eps * unit_vector and computes
    (sdf_plus - sdf_minus) / (2 * eps).

    Args:
        sdf_grid: [B, 1, D, H, W] local SDF volume.
        query_points: [B, N, 3] world-coordinate query points.
        origin: [B, 3] minimum corner of the local bounding box.
        voxel_size: scalar or [B, 1] full extent of the bounding box.
        eps: finite difference step size (default 1e-3).

    Returns:
        gradients: [B, N, 3] gradient vectors (pointing away from surface).
    """
    grad_components = []
    for axis in range(3):
        offset = torch.zeros_like(query_points)
        offset[..., axis] = eps
        sdf_plus = sample_sdf_at_points(sdf_grid, query_points + offset, origin, voxel_size)
        sdf_minus = sample_sdf_at_points(sdf_grid, query_points - offset, origin, voxel_size)
        grad_axis = (sdf_plus - sdf_minus) / (2.0 * eps)  # [B, N, 1]
        grad_components.append(grad_axis)
    return torch.cat(grad_components, dim=-1)  # [B, N, 3]


def extract_contact_points(jpos, hand_joints=None):
    """Extract hand contact query points from joint positions.

    Uses the wrist and hand joints (indices 20-23 in SMPL-H 24-joint format)
    as representative contact points for SDF querying.

    Args:
        jpos: [B, T, 24, 3] joint positions.
        hand_joints: list of int, indices of hand-related joints.
                     Default: [20, 21, 22, 23] (left wrist, left hand,
                     right wrist, right hand).

    Returns:
        contact_points: [B, T, K, 3] where K = len(hand_joints).
    """
    if hand_joints is None:
        hand_joints = [20, 21, 22, 23]
    return jpos[:, :, hand_joints, :]
