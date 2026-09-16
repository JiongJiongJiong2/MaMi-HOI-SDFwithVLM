"""Versioned hand-specific surface contact metrics."""

import math

import torch

from manip.model.sdf_utils import point_to_triangle_unsigned_distance


HAND_SURFACE_METRIC_VERSION = "hand_surface_metric_v2"
HAND_CONTACT_METRIC_VERSION = "hand_contact_common_v3"

_METRIC_KEYS = (
    "gt_proxy_left_mm",
    "pred_proxy_left_mm",
    "gt_proxy_right_mm",
    "pred_proxy_right_mm",
    "gt_hand_mesh_left_mm",
    "pred_hand_mesh_left_mm",
    "gt_hand_mesh_right_mm",
    "pred_hand_mesh_right_mm",
)

_COUNT_KEYS = (
    "left_contact_frame_count",
    "right_contact_frame_count",
)

_CONTACT_METRIC_KEYS = (
    "gt_proxy_left_mm",
    "pred_proxy_left_mm",
    "gt_hand_mesh_left_mm",
    "pred_hand_mesh_left_mm",
    "gt_proxy_right_mm",
    "pred_proxy_right_mm",
    "gt_hand_mesh_right_mm",
    "pred_hand_mesh_right_mm",
)

def _as_tensor(value, device, dtype=None):
    if torch.is_tensor(value):
        tensor = value.to(device)
    else:
        tensor = torch.as_tensor(value, device=device)
    if dtype is not None:
        tensor = tensor.to(dtype=dtype)
    return tensor


def compute_gt_proxy_contacts(
    gt_hand_jpos,
    gt_obj_verts,
    use_joints24=True,
):
    """Reproduce the existing proxy-contact labels used for grouping."""
    if gt_hand_jpos.ndim != 3 or gt_hand_jpos.shape[-1] != 3:
        raise ValueError(
            f"gt_hand_jpos must be [T, J, 3], got {gt_hand_jpos.shape}"
        )
    if gt_obj_verts.ndim != 3 or gt_obj_verts.shape[-1] != 3:
        raise ValueError(
            f"gt_obj_verts must be [T, N, 3], got {gt_obj_verts.shape}"
        )
    if gt_hand_jpos.shape[0] != gt_obj_verts.shape[0]:
        raise ValueError("GT hand and object sequences must have equal length")

    left_idx, right_idx = (22, 23) if use_joints24 else (20, 21)
    if gt_hand_jpos.shape[1] <= right_idx:
        raise ValueError(
            f"gt_hand_jpos has {gt_hand_jpos.shape[1]} joints; "
            f"joint {right_idx} is required"
        )

    device = gt_hand_jpos.device
    obj_verts = gt_obj_verts.to(device)
    threshold = 0.05 if use_joints24 else 0.10
    left_dist = torch.linalg.vector_norm(
        gt_hand_jpos[:, left_idx, None, :] - obj_verts,
        dim=-1,
    ).min(dim=1).values
    right_dist = torch.linalg.vector_norm(
        gt_hand_jpos[:, right_idx, None, :] - obj_verts,
        dim=-1,
    ).min(dim=1).values
    return left_dist < threshold, right_dist < threshold


def _mean_clearance_for_contacts(
    points,
    object_verts,
    object_faces,
    contact_mask,
    point_chunk,
    triangle_chunk,
):
    selected_frames = contact_mask.nonzero(as_tuple=False).flatten()
    if len(selected_frames) == 0:
        return None, 0

    frame_clearances = []
    for frame in selected_frames.tolist():
        triangles = object_verts[frame][object_faces]
        frame_points = points[frame]
        if frame_points.ndim == 1:
            frame_points = frame_points[None, :]
        distances, _ = point_to_triangle_unsigned_distance(
            frame_points,
            triangles,
            point_chunk=point_chunk,
            triangle_chunk=triangle_chunk,
        )
        frame_clearances.append(distances.min())

    mean_distance = torch.stack(frame_clearances).mean().item() * 1000.0
    return float(mean_distance), int(len(selected_frames))


def _frame_min_distances(
    points,
    object_verts,
    object_faces,
    frame_indices=None,
    *,
    frame_chunk=8,
    point_chunk=128,
    triangle_chunk=1024,
):
    """Return exact minimum point-to-surface distance for selected frames."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [T, P, 3], got {points.shape}")
    if object_verts.ndim != 3 or object_verts.shape[-1] != 3:
        raise ValueError(
            f"object_verts must be [T, N, 3], got {object_verts.shape}"
        )
    if points.shape[0] != object_verts.shape[0]:
        raise ValueError("points and object_verts must have equal length")
    if object_faces.ndim != 2 or object_faces.shape[-1] != 3:
        raise ValueError(
            f"object_faces must be [F, 3], got {object_faces.shape}"
        )
    if frame_chunk <= 0 or point_chunk <= 0 or triangle_chunk <= 0:
        raise ValueError("chunk sizes must be positive")

    device = points.device
    if frame_indices is None:
        frame_indices = torch.arange(points.shape[0], device=device)
    else:
        frame_indices = _as_tensor(frame_indices, device, torch.long).flatten()
    if frame_indices.numel() == 0:
        return torch.empty(0, device=device, dtype=points.dtype)
    if points.shape[1] == 0:
        raise ValueError("points must contain at least one point per frame")
    if object_faces.shape[0] == 0:
        raise ValueError("object_faces must contain at least one triangle")

    object_faces = _as_tensor(object_faces, device, torch.long)
    frame_min_distances = []
    for frame_start in range(0, frame_indices.numel(), frame_chunk):
        chunk_frames = frame_indices[
            frame_start : frame_start + frame_chunk
        ]
        chunk_points = points[chunk_frames]
        chunk_triangles = object_verts[chunk_frames][:, object_faces]
        frame_best_sq = torch.full(
            chunk_points.shape[:2],
            float("inf"),
            device=device,
            dtype=points.dtype,
        )

        for triangle_start in range(
            0,
            chunk_triangles.shape[1],
            triangle_chunk,
        ):
            triangles = chunk_triangles[
                :,
                triangle_start : triangle_start + triangle_chunk,
            ]
            a, b, c = triangles.unbind(dim=2)
            normal = torch.cross(b - a, c - a, dim=-1)
            nn = torch.einsum("bfi,bfi->bf", normal, normal)
            safe_nn = torch.where(nn > 0, nn, torch.ones_like(nn))

            for point_start in range(
                0,
                chunk_points.shape[1],
                point_chunk,
            ):
                q = chunk_points[
                    :,
                    point_start : point_start + point_chunk,
                    None,
                    :,
                ]
                signed_height = torch.einsum(
                    "bqfi,bfi->bqf",
                    q - a[:, None],
                    normal,
                ) / safe_nn[:, None]
                projection = q - signed_height[..., None] * normal[:, None]

                inside = (nn > 0)[:, None].expand(
                    -1,
                    q.shape[1],
                    -1,
                ).clone()
                for u, v in ((a, b), (b, c), (c, a)):
                    edge = (v - u)[:, None]
                    side = torch.einsum(
                        "bqfi,bqfi->bqf",
                        torch.cross(edge, projection - u[:, None], dim=-1),
                        normal[:, None],
                    )
                    inside &= side >= -1e-12 * nn[:, None]

                local_best = torch.where(
                    inside,
                    signed_height.square() * nn[:, None],
                    torch.full_like(signed_height, float("inf")),
                )
                for u, v in ((a, b), (b, c), (c, a)):
                    edge = v - u
                    ee = torch.einsum("bfi,bfi->bf", edge, edge)
                    t = torch.einsum(
                        "bqfi,bfi->bqf",
                        q - u[:, None],
                        edge,
                    ) / torch.where(
                        ee > 0,
                        ee,
                        torch.ones_like(ee),
                    )[:, None]
                    edge_closest = (
                        u[:, None]
                        + torch.clamp(t, 0.0, 1.0)[..., None]
                        * edge[:, None]
                    )
                    delta = q - edge_closest
                    d2 = torch.einsum("bqfi,bqfi->bqf", delta, delta)
                    local_best = torch.where(d2 < local_best, d2, local_best)

                chunk_best = local_best.min(dim=2).values
                frame_best_sq[
                    :,
                    point_start : point_start + point_chunk,
                ] = torch.minimum(
                    frame_best_sq[
                        :,
                        point_start : point_start + point_chunk,
                    ],
                    chunk_best,
                )

        frame_best_sq = frame_best_sq.min(dim=1).values
        positive = frame_best_sq > 0
        safe_best = torch.where(
            positive,
            frame_best_sq,
            torch.ones_like(frame_best_sq),
        )
        frame_min_distances.append(
            torch.where(
                positive,
                torch.sqrt(safe_best),
                torch.zeros_like(frame_best_sq),
            )
        )

    return torch.cat(frame_min_distances)


def _mean_frame_min_distance(
    points,
    object_verts,
    object_faces,
    frame_mask,
    point_chunk,
    triangle_chunk,
):
    """Mean over selected frames of the minimum point-to-surface distance."""
    if points.ndim != 3 or points.shape[-1] != 3:
        raise ValueError(f"points must be [T, P, 3], got {points.shape}")
    if object_verts.ndim != 3 or object_verts.shape[-1] != 3:
        raise ValueError(
            f"object_verts must be [T, N, 3], got {object_verts.shape}"
        )
    if points.shape[0] != object_verts.shape[0]:
        raise ValueError("points and object_verts must have equal length")

    selected_frames = frame_mask.nonzero(as_tuple=False).flatten()
    if len(selected_frames) == 0:
        return None, 0

    frame_distances = _frame_min_distances(
        points,
        object_verts,
        object_faces,
        selected_frames,
        point_chunk=point_chunk,
        triangle_chunk=triangle_chunk,
    )
    mean_distance = frame_distances.mean().item() * 1000.0
    return float(mean_distance), int(len(selected_frames))


def _confusion_metrics(tp, fp, tn, fn):
    precision = float(tp / (tp + fp)) if tp + fp else None
    recall = float(tp / (tp + fn)) if tp + fn else None
    f1 = (
        float((2 * tp) / (2 * tp + fp + fn))
        if 2 * tp + fp + fn
        else None
    )
    return {
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "gt_contact_frame_count": int(tp + fn),
        "pred_contact_frame_count": int(tp + fp),
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def compute_hand_contact_metrics_v3(
    gt_human_verts,
    pred_human_verts,
    gt_human_jpos,
    pred_human_jpos,
    gt_obj_verts,
    pred_obj_verts,
    object_faces,
    left_hand_vertex_idxs,
    right_hand_vertex_idxs,
    saved_left_contact,
    saved_right_contact,
    use_joints24=True,
    contact_threshold_m=0.05,
    point_chunk=256,
    triangle_chunk=1024,
):
    """Compute common hand-contact metrics from a saved GT contact mask.

    Contact prediction uses predicted joint 22/23 proxy points on every valid
    frame. Clearance metrics use GT-contact frames only. The saved mask is the
    sole primary GT source; geometry-derived masks are diagnostics only.
    """
    if contact_threshold_m <= 0:
        raise ValueError("contact_threshold_m must be positive")
    if gt_human_verts.ndim != 3 or gt_human_verts.shape[-1] != 3:
        raise ValueError(
            f"gt_human_verts must be [T, N, 3], got {gt_human_verts.shape}"
        )
    if pred_human_verts.shape != gt_human_verts.shape:
        raise ValueError(
            "pred_human_verts and gt_human_verts must have the same shape"
        )
    if gt_human_jpos.shape != pred_human_jpos.shape:
        raise ValueError("pred_human_jpos and gt_human_jpos must have the same shape")
    if gt_obj_verts.shape != pred_obj_verts.shape:
        raise ValueError("pred_obj_verts and gt_obj_verts must have the same shape")
    if object_faces.ndim != 2 or object_faces.shape[-1] != 3:
        raise ValueError(
            f"object_faces must be [F, 3], got {object_faces.shape}"
        )
    if object_faces.shape[0] == 0:
        raise ValueError("object_faces must contain at least one triangle")

    sequence_length = gt_human_verts.shape[0]
    if gt_obj_verts.shape[0] != sequence_length:
        raise ValueError("human and object sequences must have equal length")

    saved_left_contact = _as_tensor(saved_left_contact, gt_obj_verts.device, torch.bool)
    saved_right_contact = _as_tensor(
        saved_right_contact,
        gt_obj_verts.device,
        torch.bool,
    )
    if saved_left_contact.ndim != 1 or saved_right_contact.ndim != 1:
        raise ValueError("saved contact masks must be one-dimensional")
    if (
        saved_left_contact.shape[0] != sequence_length
        or saved_right_contact.shape[0] != sequence_length
    ):
        raise ValueError(
            "saved contact mask length must match the human/object sequences"
        )

    device = gt_obj_verts.device
    gt_human_verts = gt_human_verts.to(device)
    pred_human_verts = pred_human_verts.to(device)
    gt_human_jpos = gt_human_jpos.to(device)
    pred_human_jpos = pred_human_jpos.to(device)
    pred_obj_verts = pred_obj_verts.to(device)
    object_faces = _as_tensor(object_faces, device, torch.long)
    left_hand_vertex_idxs = _as_tensor(left_hand_vertex_idxs, device, torch.long)
    right_hand_vertex_idxs = _as_tensor(right_hand_vertex_idxs, device, torch.long)

    left_idx, right_idx = (22, 23) if use_joints24 else (20, 21)
    result = {
        "version": HAND_CONTACT_METRIC_VERSION,
        "valid_frame_count": int(sequence_length),
        "contact_threshold_m": float(contact_threshold_m),
    }

    for hand_name, joint_idx, vertex_idxs, saved_contact in (
        (
            "left",
            left_idx,
            left_hand_vertex_idxs,
            saved_left_contact,
        ),
        (
            "right",
            right_idx,
            right_hand_vertex_idxs,
            saved_right_contact,
        ),
    ):
        pred_proxy = pred_human_jpos[:, joint_idx][:, None, :]
        pred_proxy_distances = _frame_min_distances(
            pred_proxy,
            pred_obj_verts,
            object_faces,
            point_chunk=point_chunk,
            triangle_chunk=triangle_chunk,
        )
        pred_contact = pred_proxy_distances < contact_threshold_m

        tp = int((saved_contact & pred_contact).sum().item())
        fp = int(((~saved_contact) & pred_contact).sum().item())
        tn = int(((~saved_contact) & (~pred_contact)).sum().item())
        fn = int((saved_contact & (~pred_contact)).sum().item())
        result.update(
            {
                f"{hand_name}_{key}": value
                for key, value in _confusion_metrics(tp, fp, tn, fn).items()
            }
        )

        gt_proxy_mean, gt_proxy_count = _mean_frame_min_distance(
            gt_human_jpos[:, joint_idx][:, None, :],
            gt_obj_verts,
            object_faces,
            saved_contact,
            point_chunk,
            triangle_chunk,
        )
        pred_proxy_mean = (
            float(pred_proxy_distances[saved_contact].mean().item() * 1000.0)
            if int(saved_contact.sum().item())
            else None
        )
        gt_mesh_mean, gt_mesh_count = _mean_frame_min_distance(
            gt_human_verts[:, vertex_idxs],
            gt_obj_verts,
            object_faces,
            saved_contact,
            point_chunk,
            triangle_chunk,
        )
        pred_mesh_mean, pred_mesh_count = _mean_frame_min_distance(
            pred_human_verts[:, vertex_idxs],
            pred_obj_verts,
            object_faces,
            saved_contact,
            point_chunk,
            triangle_chunk,
        )
        if len({gt_proxy_count, gt_mesh_count, pred_mesh_count}) != 1:
            raise RuntimeError(
                "GT-contact frame counts differ across clearance metrics"
            )

        result[f"gt_proxy_{hand_name}_mm"] = gt_proxy_mean
        result[f"pred_proxy_{hand_name}_mm"] = pred_proxy_mean
        result[f"gt_hand_mesh_{hand_name}_mm"] = gt_mesh_mean
        result[f"pred_hand_mesh_{hand_name}_mm"] = pred_mesh_mean
        result[f"{hand_name}_contact_frame_count"] = int(
            saved_contact.sum().item()
        )

    return result


def aggregate_hand_contact_metrics_v3(records):
    """Aggregate v3 metrics with micro contact counts and macro clearances."""
    records = list(records)
    aggregate = {
        "version": HAND_CONTACT_METRIC_VERSION,
        "sequence_count": len(records),
        "valid_frame_count": int(
            sum(int(record.get("valid_frame_count", 0)) for record in records)
        ),
        "micro": {},
        "macro": {},
        "clearance_macro_mm": {},
    }

    for hand_name in ("left", "right"):
        totals = {
            key: int(sum(int(record.get(f"{hand_name}_{key}", 0)) for record in records))
            for key in ("tp", "fp", "tn", "fn")
        }
        aggregate["micro"][hand_name] = _confusion_metrics(**totals)

        macro = {}
        for metric in ("precision", "recall", "f1"):
            values = [
                float(record[f"{hand_name}_{metric}"])
                for record in records
                if record.get(f"{hand_name}_{metric}") is not None
            ]
            macro[metric] = (
                float(math.fsum(values) / len(values)) if values else None
            )
            macro[f"{metric}_valid_sequence_count"] = len(values)
        aggregate["macro"][hand_name] = macro

        clearance = {}
        for metric in _CONTACT_METRIC_KEYS:
            if not metric.endswith(f"{hand_name}_mm"):
                continue
            values = [
                float(record[metric])
                for record in records
                if record.get(metric) is not None
            ]
            clearance[metric] = (
                float(math.fsum(values) / len(values)) if values else None
            )
            clearance[f"{metric}_valid_sequence_count"] = len(values)
        aggregate["clearance_macro_mm"][hand_name] = clearance

    return aggregate


def compute_hand_surface_metrics_v2(
    gt_human_verts,
    pred_human_verts,
    gt_human_jpos,
    pred_human_jpos,
    gt_obj_verts,
    pred_obj_verts,
    object_faces,
    left_hand_vertex_idxs,
    right_hand_vertex_idxs,
    gt_left_contact,
    gt_right_contact,
    use_joints24=True,
    point_chunk=256,
    triangle_chunk=1024,
):
    """Compute per-hand proxy and hand-mesh clearances on GT contact frames.

    All distances are unsigned and expressed in millimetres. A zero clearance
    is preserved as a valid measurement; ``None`` is used only when the
    corresponding GT-contact frame set is empty.
    """
    if gt_human_verts.ndim != 3 or gt_human_verts.shape[-1] != 3:
        raise ValueError(
            f"gt_human_verts must be [T, N, 3], got {gt_human_verts.shape}"
        )
    if pred_human_verts.shape != gt_human_verts.shape:
        raise ValueError(
            "pred_human_verts and gt_human_verts must have the same shape"
        )
    if gt_human_jpos.shape != pred_human_jpos.shape:
        raise ValueError("pred_human_jpos and gt_human_jpos must have the same shape")
    if gt_obj_verts.shape != pred_obj_verts.shape:
        raise ValueError("pred_obj_verts and gt_obj_verts must have the same shape")
    if object_faces.ndim != 2 or object_faces.shape[-1] != 3:
        raise ValueError(
            f"object_faces must be [F, 3], got {object_faces.shape}"
        )
    if object_faces.shape[0] == 0:
        raise ValueError("object_faces must contain at least one triangle")

    device = gt_obj_verts.device
    gt_human_verts = gt_human_verts.to(device)
    pred_human_verts = pred_human_verts.to(device)
    gt_human_jpos = gt_human_jpos.to(device)
    pred_human_jpos = pred_human_jpos.to(device)
    pred_obj_verts = pred_obj_verts.to(device)
    object_faces = _as_tensor(object_faces, device, torch.long)
    left_hand_vertex_idxs = _as_tensor(left_hand_vertex_idxs, device, torch.long)
    right_hand_vertex_idxs = _as_tensor(right_hand_vertex_idxs, device, torch.long)
    gt_left_contact = _as_tensor(gt_left_contact, device, torch.bool)
    gt_right_contact = _as_tensor(gt_right_contact, device, torch.bool)

    left_idx, right_idx = (22, 23) if use_joints24 else (20, 21)
    hands = (
        ("left", left_idx, left_hand_vertex_idxs, gt_left_contact),
        ("right", right_idx, right_hand_vertex_idxs, gt_right_contact),
    )

    result = {
        "version": HAND_SURFACE_METRIC_VERSION,
    }
    for hand_name, joint_idx, vertex_idxs, contact_mask in hands:
        for source_name, human_verts, human_jpos, obj_verts in (
            ("gt", gt_human_verts, gt_human_jpos, gt_obj_verts),
            ("pred", pred_human_verts, pred_human_jpos, pred_obj_verts),
        ):
            proxy_distance, proxy_count = _mean_clearance_for_contacts(
                human_jpos[:, joint_idx],
                obj_verts,
                object_faces,
                contact_mask,
                point_chunk,
                triangle_chunk,
            )
            mesh_distance, mesh_count = _mean_clearance_for_contacts(
                human_verts[:, vertex_idxs],
                obj_verts,
                object_faces,
                contact_mask,
                point_chunk,
                triangle_chunk,
            )
            if proxy_count != mesh_count:
                raise RuntimeError(
                    "Proxy and hand-mesh metrics selected different frame counts"
                )
            result[f"{source_name}_proxy_{hand_name}_mm"] = proxy_distance
            result[f"{source_name}_hand_mesh_{hand_name}_mm"] = mesh_distance
        result[f"{hand_name}_contact_frame_count"] = int(contact_mask.sum().item())

    return result


def aggregate_hand_surface_metrics_v2(records):
    """Macro-average sequence metrics while preserving valid zero distances."""
    records = list(records)
    aggregate = {
        "version": HAND_SURFACE_METRIC_VERSION,
        "sequence_count": len(records),
    }
    for key in _METRIC_KEYS:
        values = [
            float(record[key])
            for record in records
            if record.get(key) is not None
        ]
        aggregate[key] = (
            float(math.fsum(values) / len(values)) if values else None
        )
        aggregate[f"{key}_valid_sequence_count"] = len(values)
    for key in _COUNT_KEYS:
        aggregate[key] = int(sum(int(record.get(key, 0)) for record in records))
    return aggregate
