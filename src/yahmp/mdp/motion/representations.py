"""Tensor-serialization helpers for YAHMP motion command representations."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
  quat_apply_inverse,
)

from .library import MotionFrameBatch


def joint_ref_anchor_rp_representation(
  joint_pos: torch.Tensor,
  joint_vel: torch.Tensor,
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  anchor_lin_vel_w: torch.Tensor,
  anchor_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
  """Serialize current joint refs with anchor motion terms and roll/pitch."""
  anchor_lin_vel_b = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
  anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
  roll, pitch, _ = _quat_roll_pitch_yaw(anchor_quat_w)
  return torch.cat(
    (
      joint_pos,
      joint_vel,
      anchor_lin_vel_b[..., :2],
      anchor_ang_vel_b[..., 2:3],
      anchor_pos_w[..., 2:3],
      roll[..., None],
      pitch[..., None],
    ),
    dim=-1,
  )


def joint_ref_only_anchor_rp_representation(
  joint_pos: torch.Tensor,
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  anchor_lin_vel_w: torch.Tensor,
  anchor_ang_vel_w: torch.Tensor,
) -> torch.Tensor:
  """Serialize current joint refs without joint velocities."""
  anchor_lin_vel_b = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
  anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)
  roll, pitch, _ = _quat_roll_pitch_yaw(anchor_quat_w)
  return torch.cat(
    (
      joint_pos,
      anchor_lin_vel_b[..., :2],
      anchor_ang_vel_b[..., 2:3],
      anchor_pos_w[..., 2:3],
      roll[..., None],
      pitch[..., None],
    ),
    dim=-1,
  )


def future_joint_ref_anchor_rp_representation(frames: MotionFrameBatch) -> torch.Tensor:
  """Serialize future joint refs with anchor motion terms and roll/pitch."""
  anchor_lin_vel_b = quat_apply_inverse(frames.anchor_quat_w, frames.anchor_lin_vel_w)
  anchor_ang_vel_b = quat_apply_inverse(frames.anchor_quat_w, frames.anchor_ang_vel_w)
  roll, pitch, _ = _quat_roll_pitch_yaw(frames.anchor_quat_w)
  return torch.cat(
    [
      frames.joint_pos,
      frames.joint_vel,
      anchor_lin_vel_b[..., :2],
      anchor_ang_vel_b[..., 2:3],
      frames.anchor_pos_w[..., 2:3],
      roll[..., None],
      pitch[..., None],
    ],
    dim=-1,
  )


def _quat_roll_pitch_yaw(quat_wxyz: torch.Tensor) -> tuple[torch.Tensor, ...]:
  w, x, y, z = quat_wxyz.unbind(dim=-1)

  sinr_cosp = 2.0 * (w * x + y * z)
  cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
  roll = torch.atan2(sinr_cosp, cosr_cosp)

  sinp = 2.0 * (w * y - z * x)
  pitch = torch.asin(torch.clamp(sinp, -1.0, 1.0))

  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  yaw = torch.atan2(siny_cosp, cosy_cosp)
  return roll, pitch, yaw
