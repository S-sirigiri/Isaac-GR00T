from __future__ import annotations

import math
from typing import Callable

import torch

from gr00t.configs.l_obstacle.config import load as _load_l_obstacle_config


def _resolve_pair(value, fallback):
    if value is None:
        return fallback
    return (float(value[0]), float(value[1]))


def _resolve_constraint_params() -> dict:
    """Pulls obstacle + robot-box params from the SoT YAML on every call.

    Reads through ``gr00t.configs.l_obstacle.config.load`` which mtime-caches
    the YAML, so hot edits propagate without restarting the server.
    """
    cfg = _load_l_obstacle_config()
    constraint_cfg = cfg.get("constraint", {}) or {}

    return {
        "position_xy": _resolve_pair(cfg.get("position_xy"), (0.0, 0.0)),
        "arm_length": float(cfg.get("arm_length", 0.10)),
        "arm_thickness": float(cfg.get("arm_thickness", 0.02)),
        "z_rotation_rad": math.radians(float(cfg.get("z_rotation_deg", 0.0))),
        "epsilon": float(constraint_cfg.get("epsilon", 0.01)),
        "ee_halfsize_xy": _resolve_pair(constraint_cfg.get("ee_halfsize_xy"), (0.040, 0.040)),
        "ee_offset_xy": _resolve_pair(constraint_cfg.get("ee_offset_xy"), (0.0, 0.0)),
        "left_gripper_halfsize_xy": _resolve_pair(
            constraint_cfg.get("left_gripper_halfsize_xy"), (0.012, 0.025)
        ),
        "left_gripper_offset_xy": _resolve_pair(
            constraint_cfg.get("left_gripper_offset_xy"), (-0.035, 0.0)
        ),
        "right_gripper_halfsize_xy": _resolve_pair(
            constraint_cfg.get("right_gripper_halfsize_xy"), (0.012, 0.025)
        ),
        "right_gripper_offset_xy": _resolve_pair(
            constraint_cfg.get("right_gripper_offset_xy"), (0.035, 0.0)
        ),
        "butter_halfsize_xy": _resolve_pair(
            constraint_cfg.get("butter_halfsize_xy"), (0.00871, 0.01977)
        ),
        "butter_offset_xy": _resolve_pair(constraint_cfg.get("butter_offset_xy"), (0.0, 0.0)),
    }


def _robot_aabb_xy(
    position: torch.Tensor,
    offset_xy: tuple[float, float],
    halfsize_xy: tuple[float, float],
    as_like: Callable[[float], torch.Tensor],
):
    cx = position[..., 0] + as_like(offset_xy[0])
    cy = position[..., 1] + as_like(offset_xy[1])
    hx = as_like(halfsize_xy[0])
    hy = as_like(halfsize_xy[1])
    return cx - hx, cx + hx, cy - hy, cy + hy


def _l_arm_obbs(params: dict, as_like: Callable[[float], torch.Tensor]):
    """Two OBBs (one per L arm) in world frame.

    Each OBB is returned as ``(center_x, center_y, half_u, half_v, u_axis, v_axis)``
    where u_axis is the rotated local-x axis and v_axis is the rotated local-y
    axis. The L corner is at ``position_xy``; arms extend along the rotated
    +u and +v directions.
    """
    corner_x, corner_y = params["position_xy"]
    arm_length = params["arm_length"]
    arm_thickness = params["arm_thickness"]
    half_length = arm_length / 2.0
    half_thick = arm_thickness / 2.0

    cos_t = math.cos(params["z_rotation_rad"])
    sin_t = math.sin(params["z_rotation_rad"])
    # u = R(theta) @ (1, 0); v = R(theta) @ (0, 1)
    u_axis = (as_like(cos_t), as_like(sin_t))
    v_axis = (as_like(-sin_t), as_like(cos_t))

    # Horizontal arm: local center (half_length, 0); half-extents (half_length, half_thick).
    horz_center = (
        as_like(corner_x + cos_t * half_length),
        as_like(corner_y + sin_t * half_length),
    )
    horz_half_u = as_like(half_length)
    horz_half_v = as_like(half_thick)

    # Vertical arm: local center (0, half_length); half-extents (half_thick, half_length).
    vert_center = (
        as_like(corner_x - sin_t * half_length),
        as_like(corner_y + cos_t * half_length),
    )
    vert_half_u = as_like(half_thick)
    vert_half_v = as_like(half_length)

    return (
        (horz_center, horz_half_u, horz_half_v, u_axis, v_axis),
        (vert_center, vert_half_u, vert_half_v, u_axis, v_axis),
    )


def _aabb_obb_signed_separation(aabb, obb, as_like) -> torch.Tensor:
    """Signed gap in metres between an AABB and an OBB in xy (SAT).

    Returns a tensor matching the AABB's batch shape. Positive when disjoint,
    zero when touching, negative when overlapping (most-negative dimension is
    the penetration depth).
    """
    aabb_min_x, aabb_max_x, aabb_min_y, aabb_max_y = aabb
    obb_center, obb_half_u, obb_half_v, u_axis, v_axis = obb

    aabb_center_x = (aabb_min_x + aabb_max_x) / 2.0
    aabb_center_y = (aabb_min_y + aabb_max_y) / 2.0
    aabb_half_x = (aabb_max_x - aabb_min_x) / 2.0
    aabb_half_y = (aabb_max_y - aabb_min_y) / 2.0

    dx = obb_center[0] - aabb_center_x
    dy = obb_center[1] - aabb_center_y

    axes = (
        (as_like(1.0), as_like(0.0)),  # world x
        (as_like(0.0), as_like(1.0)),  # world y
        u_axis,
        v_axis,
    )

    gaps = []
    for ax_x, ax_y in axes:
        aabb_proj = aabb_half_x * torch.abs(ax_x) + aabb_half_y * torch.abs(ax_y)
        u_proj = torch.abs(ax_x * u_axis[0] + ax_y * u_axis[1])
        v_proj = torch.abs(ax_x * v_axis[0] + ax_y * v_axis[1])
        obb_proj = obb_half_u * u_proj + obb_half_v * v_proj
        center_sep = torch.abs(ax_x * dx + ax_y * dy)
        gap = center_sep - aabb_proj - obb_proj
        gaps.append(gap)
    return torch.stack(gaps, dim=-1).max(dim=-1).values


def l_obstacle_avoidance_penalty(
    position: torch.Tensor,
    as_like: Callable[[float], torch.Tensor],
    epsilon: float | None = None,
) -> torch.Tensor:
    """Squared-hinge penalty enforcing AABB-vs-rotated-L separation in xy.

    Args:
        position: ``[..., 3]`` world-frame EE positions along the proxy
            trajectory. Only xy is used.
        as_like: callable returning a scalar tensor in the dtype/device of
            ``position``.
        epsilon: optional override for the clearance margin (metres). When
            ``None`` (default), reads from the YAML config.

    Returns:
        Scalar tensor — ``sum_{(box, arm)} clamp(eps - signed_separation, 0)^2``,
        summed across the trajectory steps. Same squared-hinge shape as
        ``CostConstraintObjective._cylinder_avoidance_penalty_squared_hinge``.
    """
    params = _resolve_constraint_params()
    eps = as_like(epsilon if epsilon is not None else params["epsilon"])

    robot_boxes = (
        _robot_aabb_xy(position, params["ee_offset_xy"], params["ee_halfsize_xy"], as_like),
        _robot_aabb_xy(
            position, params["left_gripper_offset_xy"], params["left_gripper_halfsize_xy"], as_like
        ),
        _robot_aabb_xy(
            position,
            params["right_gripper_offset_xy"],
            params["right_gripper_halfsize_xy"],
            as_like,
        ),
        _robot_aabb_xy(
            position, params["butter_offset_xy"], params["butter_halfsize_xy"], as_like
        ),
    )
    obstacle_obbs = _l_arm_obbs(params, as_like)

    total = position.new_zeros(())
    for robot_box in robot_boxes:
        for obstacle_obb in obstacle_obbs:
            separation = _aabb_obb_signed_separation(robot_box, obstacle_obb, as_like)
            violation = torch.clamp(eps - separation, min=0.0)
            total = total + violation.square().sum()
    return total
