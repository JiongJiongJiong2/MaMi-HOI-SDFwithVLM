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


def point_to_triangle_unsigned_distance(
    points,
    triangles,
    point_chunk=32,
    triangle_chunk=256,
):
    """Return exact unsigned point-to-triangle distances with gradients.

    The implementation follows the same closed-plane plus closed-edge
    formulation as the CPU diagnostic. The returned closest point is the
    selected branch; medial-axis ties are not treated as a single smooth
    function.
    """
    points = points.reshape(-1, 3)
    triangles = triangles.reshape(-1, 3, 3)
    device = points.device
    dtype = points.dtype
    distances = []
    closest_points = []

    for point_start in range(0, len(points), point_chunk):
        q = points[point_start : point_start + point_chunk]
        best = torch.full(
            (len(q),),
            float("inf"),
            device=device,
            dtype=dtype,
        )
        best_closest = torch.zeros_like(q)

        for triangle_start in range(0, len(triangles), triangle_chunk):
            tri = triangles[
                triangle_start : triangle_start + triangle_chunk
            ]
            a, b, c = tri[:, 0], tri[:, 1], tri[:, 2]
            normal = torch.cross(b - a, c - a, dim=-1)
            nn = torch.einsum("ki,ki->k", normal, normal)
            safe_nn = torch.where(nn > 0, nn, torch.ones_like(nn))
            signed_height = torch.einsum(
                "qki,ki->qk", q[:, None] - a, normal
            ) / safe_nn
            projection = q[:, None] - signed_height[..., None] * normal

            inside = (nn > 0).unsqueeze(0).expand_as(signed_height).clone()
            for u, v in ((a, b), (b, c), (c, a)):
                edge = (v - u).unsqueeze(0)
                side = torch.einsum(
                    "qki,ki->qk",
                    torch.cross(edge, projection - u, dim=-1),
                    normal,
                )
                inside &= side >= -1e-12 * nn

            local_best = torch.where(
                inside,
                signed_height.square() * nn,
                torch.full_like(signed_height, float("inf")),
            )
            local_closest = projection.clone()

            for u, v in ((a, b), (b, c), (c, a)):
                edge = v - u
                ee = torch.einsum("ki,ki->k", edge, edge)
                t = torch.einsum(
                    "qki,ki->qk", q[:, None] - u, edge
                ) / torch.where(ee > 0, ee, torch.ones_like(ee))
                edge_closest = u + torch.clamp(t, 0.0, 1.0)[..., None] * edge
                delta = q[:, None] - edge_closest
                d2 = torch.einsum("qki,qki->qk", delta, delta)
                improve = d2 < local_best
                local_best = torch.where(improve, d2, local_best)
                local_closest = torch.where(
                    improve[..., None],
                    edge_closest,
                    local_closest,
                )

            selected = local_best.argmin(dim=1)
            value = local_best.gather(
                1, selected[:, None]
            ).squeeze(1)
            selected_cp = local_closest.gather(
                1,
                selected[:, None, None].expand(-1, -1, 3),
            ).squeeze(1)
            improve = value < best
            best = torch.where(improve, value, best)
            best_closest = torch.where(
                improve[:, None],
                selected_cp,
                best_closest,
            )

        distances.append(torch.sqrt(best))
        closest_points.append(best_closest)

    return torch.cat(distances), torch.cat(closest_points)


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


def project_to_so3(rotation_matrices, eps=1e-6):
    """Map predicted 3x3 matrices to proper rotations for geometry queries.

    This uses the continuous 6D/Gram-Schmidt construction on the first two
    rows.  It avoids the repeated-singular-value gradient ambiguity of an SVD
    projection, preserves an already-valid rotation, and guarantees det=+1.
    Collapsed or collinear inputs fail fast instead of producing invalid query
    geometry.
    """
    if rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation_matrices must end in [3, 3], "
            f"got {rotation_matrices.shape}"
        )
    if not torch.isfinite(rotation_matrices).all():
        raise ValueError("rotation_matrices contains NaN or Inf")

    work = (
        rotation_matrices.float()
        if rotation_matrices.dtype in (torch.float16, torch.bfloat16)
        else rotation_matrices
    )
    first = work[..., 0, :]
    second = work[..., 1, :]
    first_norm = torch.linalg.vector_norm(first, dim=-1, keepdim=True)
    first_unit = first / first_norm.clamp_min(eps)
    second_orthogonal = second - (
        first_unit * (first_unit * second).sum(dim=-1, keepdim=True)
    )
    second_norm = torch.linalg.vector_norm(
        second_orthogonal, dim=-1, keepdim=True
    )
    if (first_norm < eps).any() or (second_norm < eps).any():
        raise ValueError(
            "rotation_matrices contains collapsed or collinear basis rows"
        )
    second_unit = second_orthogonal / second_norm
    third_unit = torch.linalg.cross(first_unit, second_unit, dim=-1)
    return torch.stack((first_unit, second_unit, third_unit), dim=-2)


def rotation_validity_statistics(rotation_matrices):
    """Return per-matrix orthogonality error and determinant."""
    if rotation_matrices.shape[-2:] != (3, 3):
        raise ValueError(
            "rotation_matrices must end in [3, 3], "
            f"got {rotation_matrices.shape}"
        )
    work = (
        rotation_matrices.float()
        if rotation_matrices.dtype in (torch.float16, torch.bfloat16)
        else rotation_matrices
    )
    identity = torch.eye(3, device=work.device, dtype=work.dtype)
    gram = torch.matmul(work.transpose(-1, -2), work)
    orthogonality_error = torch.linalg.matrix_norm(
        gram - identity, ord="fro", dim=(-2, -1)
    )
    return orthogonality_error, torch.det(work)


def object_to_world_points(points_object, object_rot_mat, object_com_pos):
    """Transform row-vector canonical points to world coordinates."""
    if points_object.ndim != 4 or points_object.shape[-1] != 3:
        raise ValueError(
            f"points_object must be [B, T, K, 3], got {points_object.shape}"
        )
    if object_rot_mat.shape[-2:] != (3, 3):
        raise ValueError(f"object_rot_mat must end in [3, 3], got {object_rot_mat.shape}")
    if object_com_pos.shape[-1] != 3:
        raise ValueError(f"object_com_pos must end in 3, got {object_com_pos.shape}")
    return (
        torch.matmul(points_object, object_rot_mat.transpose(-1, -2))
        + object_com_pos[:, :, None, :]
    )


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


def build_dynamic_sdf_prediction_query(
    predicted_palm_world,
    predicted_object_rotation,
    predicted_object_com,
):
    """Build the U1 geometry query exclusively from predicted quantities."""
    rotation_for_query = project_to_so3(predicted_object_rotation)
    query_points_object = world_to_object_points(
        predicted_palm_world.to(rotation_for_query.dtype),
        rotation_for_query,
        predicted_object_com.to(rotation_for_query.dtype),
    )
    return query_points_object, rotation_for_query


def object_sdf_in_bounds_mask(query_points_object, centroid, extents, atol=1e-6):
    """Return whether canonical query points fall inside the sampled SDF cube.

    MaMi-HOI normalizes all axes by the largest full bounding-box extent, so
    the valid grid domain is the cube ``centroid +/- max(extents)/2``.
    """
    if query_points_object.ndim != 3 or query_points_object.shape[-1] != 3:
        raise ValueError(
            "query_points_object must be [B, N, 3], "
            f"got {query_points_object.shape}"
        )
    if centroid.shape != (query_points_object.shape[0], 3):
        raise ValueError(f"centroid must be [B, 3], got {centroid.shape}")
    if extents.shape != (query_points_object.shape[0], 3):
        raise ValueError(f"extents must be [B, 3], got {extents.shape}")
    if not torch.isfinite(query_points_object).all():
        raise ValueError("query_points_object contains NaN or Inf")
    if not torch.isfinite(centroid).all() or not torch.isfinite(extents).all():
        raise ValueError("centroid/extents contains NaN or Inf")
    if (extents <= 0).any():
        raise ValueError("extents must be strictly positive")

    max_extent = extents.max(dim=-1, keepdim=True).values
    query_norm = (
        (query_points_object - centroid[:, None, :])
        * 2.0
        / max_extent[:, None, :]
    )
    return (query_norm.abs() <= 1.0 + atol).all(dim=-1)


def sample_object_sdf_at_points(
    sdf_grid,
    query_points_object,
    centroid,
    extents,
    return_valid_mask=False,
):
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

    if not torch.isfinite(sdf_grid).all():
        raise ValueError("sdf_grid contains NaN or Inf")
    valid_mask = object_sdf_in_bounds_mask(
        query_points_object, centroid, extents
    )
    max_extent = extents.max(dim=-1, keepdim=True).values
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
    signed_distances = sampled * (max_extent[:, None, :] / 2.0)
    if return_valid_mask:
        return signed_distances, valid_mask
    return signed_distances


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
