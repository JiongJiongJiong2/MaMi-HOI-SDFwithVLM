"""
SDF (Signed Distance Field) utility functions for MaMi-HOI.

Provides batched, differentiable SDF queries on precomputed local SDF volumes,
gradient computation via finite differences, hand contact point extraction, and
the dynamic object-SDF losses used by the lightweight experiment pipeline.

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


def world_to_object_points(points_world, object_rot_mat, object_com_pos):
    """Transform world-space points into the object's canonical coordinate frame.

    Args:
        points_world: [B, T, K, 3].
        object_rot_mat: [B, T, 3, 3], mapping object-local column vectors to
            world-space column vectors.
        object_com_pos: [B, T, 3].

    Returns:
        points_object: [B, T, K, 3].

    Notes:
        This is intentionally separate from the legacy ``origin/voxel_size``
        query path above. Dynamic training uses explicit ``centroid/extents``
        metadata to avoid the old coordinate-convention ambiguity.
    """
    if points_world.ndim != 4 or points_world.shape[-1] != 3:
        raise ValueError(f"points_world must be [B, T, K, 3], got {points_world.shape}")
    if object_rot_mat.shape[-2:] != (3, 3):
        raise ValueError(f"object_rot_mat must end in [3, 3], got {object_rot_mat.shape}")
    if object_com_pos.shape[-1] != 3:
        raise ValueError(f"object_com_pos must end in 3, got {object_com_pos.shape}")

    # For row-vector points, (p_world - t)^T R equals R^T (p_world - t)
    # in the conventional column-vector notation.
    relative_points = points_world - object_com_pos[:, :, None, :]
    return torch.matmul(relative_points, object_rot_mat)


def sample_object_sdf_at_points(sdf_grid, query_points_object, centroid, extents):
    """Differentiably query canonical object SDFs at object-local points.

    The metadata convention matches the original MaMi-HOI evaluator:
    ``centroid`` is the centre of the SDF bounding box, ``extents`` contains
    its three full side lengths, and stored SDF values are normalized by half
    of the largest extent. The return value is therefore in world units.

    Args:
        sdf_grid: [B, 1, D, H, W].
        query_points_object: [B, N, 3] in canonical object coordinates.
        centroid: [B, 3].
        extents: [B, 3].

    Returns:
        signed_distances: [B, N, 1] in world units.
    """
    if sdf_grid.ndim != 5 or sdf_grid.shape[1] != 1:
        raise ValueError(f"sdf_grid must be [B, 1, D, H, W], got {sdf_grid.shape}")
    if query_points_object.ndim != 3 or query_points_object.shape[-1] != 3:
        raise ValueError(
            "query_points_object must be [B, N, 3], "
            f"got {query_points_object.shape}"
        )
    if centroid.shape != (sdf_grid.shape[0], 3):
        raise ValueError(f"centroid must be [B, 3], got {centroid.shape}")
    if extents.shape != (sdf_grid.shape[0], 3):
        raise ValueError(f"extents must be [B, 3], got {extents.shape}")

    max_extent = extents.max(dim=-1, keepdim=True).values.clamp_min(1e-6)
    query_norm = (query_points_object - centroid[:, None, :]) * 2.0 / max_extent[:, None, :]
    # PyTorch 3D grid_sample expects coordinates in W/H/D order.
    query_norm = query_norm[..., [2, 1, 0]]
    query_grid = query_norm[:, None, None, :, :]

    sampled = F.grid_sample(
        sdf_grid,
        query_grid,
        mode='bilinear',
        padding_mode='border',
        align_corners=True,
    )
    sampled = sampled.squeeze(2).squeeze(2).permute(0, 2, 1)
    return sampled * (max_extent[:, None, :] / 2.0)


def masked_mean(values, mask, eps=1e-8):
    """Mean over a broadcastable mask, returning zero for an empty mask."""
    mask = mask.to(dtype=values.dtype)
    while mask.ndim < values.ndim:
        mask = mask.unsqueeze(-1)
    expanded_mask = mask.expand_as(values)
    denominator = expanded_mask.sum()
    if denominator.detach().item() == 0:
        return values.new_zeros(())
    return (values * expanded_mask).sum() / denominator.clamp_min(eps)


def sdf_contact_losses(pred_sdf_normalized, contact_mask, valid_mask):
    """Return penetration and surface-contact losses for dynamic SDF queries.

    ``pred_sdf_normalized`` is signed distance divided by the object's largest
    bounding-box extent. Penetration is penalised on every valid frame, while
    attraction to the surface is only applied at labelled hand-contact frames.
    """
    penetration_loss = masked_mean(F.relu(-pred_sdf_normalized), valid_mask)
    contact_loss = masked_mean(pred_sdf_normalized.abs(), valid_mask * contact_mask)
    return penetration_loss, contact_loss


def _masked_smooth_l1_per_sample(prediction, target, mask, beta=0.02, eps=1e-8):
    """Contact-masked Smooth-L1 distance for every batch element."""
    if prediction.shape != target.shape:
        raise ValueError(f"prediction and target shapes differ: {prediction.shape} vs {target.shape}")
    mask = mask.to(dtype=prediction.dtype)
    while mask.ndim < prediction.ndim:
        mask = mask.unsqueeze(-1)
    mask = mask.expand_as(prediction)
    distance = F.smooth_l1_loss(prediction, target, reduction='none', beta=beta)
    reduce_dims = tuple(range(1, distance.ndim))
    numerator = (distance * mask).sum(dim=reduce_dims)
    denominator = mask.sum(dim=reduce_dims)
    return numerator / denominator.clamp_min(eps), denominator > 0


def sdf_trajectory_ranking_loss(
    pred_sdf_normalized,
    gt_sdf_normalized,
    contact_mask,
    valid_mask,
    penetration_value=-0.03,
    floating_value=0.10,
    margin=0.02,
):
    """Lightweight contrastive ranking on signed-distance trajectories.

    The predicted trajectory is pulled closer to the GT trajectory than to two
    synthetic but physically implausible contact trajectories: one inside the
    object and one floating away from its surface. No encoder, batch negatives,
    or temperature parameter is required.
    """
    if pred_sdf_normalized.shape != gt_sdf_normalized.shape:
        raise ValueError(
            "pred_sdf_normalized and gt_sdf_normalized must share shape, got "
            f"{pred_sdf_normalized.shape} and {gt_sdf_normalized.shape}"
        )

    contact_valid_mask = valid_mask * contact_mask
    penetration_target = torch.where(
        contact_valid_mask.bool(),
        torch.full_like(gt_sdf_normalized, penetration_value),
        gt_sdf_normalized,
    )
    floating_target = torch.where(
        contact_valid_mask.bool(),
        torch.full_like(gt_sdf_normalized, floating_value),
        gt_sdf_normalized,
    )

    d_pos, has_contact = _masked_smooth_l1_per_sample(
        pred_sdf_normalized, gt_sdf_normalized, contact_valid_mask
    )
    d_pen, _ = _masked_smooth_l1_per_sample(
        pred_sdf_normalized, penetration_target, contact_valid_mask
    )
    d_float, _ = _masked_smooth_l1_per_sample(
        pred_sdf_normalized, floating_target, contact_valid_mask
    )

    ranking = F.relu(margin + d_pos - d_pen) + F.relu(margin + d_pos - d_float)
    if not has_contact.any():
        return pred_sdf_normalized.new_zeros(())
    return ranking[has_contact].mean()
