"""Tensor-serialization helpers for YAHMP motion command representations."""

from __future__ import annotations

import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
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


def _rot6d_from_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
  """Encode orientation as the first two rotation-matrix columns."""
  rotm = matrix_from_quat(quat_wxyz)
  return rotm[..., :, :2].transpose(-2, -1).reshape(*rotm.shape[:-2], 6)


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


def hand_base_representation(
  body_pos_w: torch.Tensor,
  body_quat_w: torch.Tensor,
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  anchor_lin_vel_w: torch.Tensor,
  anchor_ang_vel_w: torch.Tensor,
  left_hand_body_index: int,
  right_hand_body_index: int,
) -> torch.Tensor:
  """Serialize current hand poses with anchor planar velocity and height."""
  hand_pos_w = torch.stack(
    (
      body_pos_w[:, left_hand_body_index],
      body_pos_w[:, right_hand_body_index],
    ),
    dim=1,
  )
  hand_quat_w = torch.stack(
    (
      body_quat_w[:, left_hand_body_index],
      body_quat_w[:, right_hand_body_index],
    ),
    dim=1,
  )

  hand_anchor_pos_w = anchor_pos_w[:, None, :].expand(-1, 2, -1)
  hand_anchor_quat_w = anchor_quat_w[:, None, :].expand(-1, 2, -1)
  hand_pos_b, hand_quat_b = subtract_frame_transforms(
    hand_anchor_pos_w,
    hand_anchor_quat_w,
    hand_pos_w,
    hand_quat_w,
  )
  hand_rot6d_b = _rot6d_from_quat(hand_quat_b)

  anchor_lin_vel_b = quat_apply_inverse(anchor_quat_w, anchor_lin_vel_w)
  anchor_ang_vel_b = quat_apply_inverse(anchor_quat_w, anchor_ang_vel_w)

  return torch.cat(
    (
      hand_pos_b.reshape(body_pos_w.shape[0], -1),
      hand_rot6d_b.reshape(body_pos_w.shape[0], -1),
      anchor_lin_vel_b[:, :2],
      anchor_ang_vel_b[:, 2:3],
      anchor_pos_w[:, 2:3],
    ),
    dim=-1,
  )
