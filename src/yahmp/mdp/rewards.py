from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .motion.base import MotionCommand
from .motion.hand_base import HandBaseMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def motion_joint_position_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.joint_pos - command.robot_joint_pos)
  return torch.exp(-error.mean(dim=-1) / std**2)


def motion_joint_velocity_error_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  error = torch.square(command.joint_vel - command.robot_joint_vel)
  return torch.exp(-error.mean(dim=-1) / std**2)


def _hand_base_command(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> HandBaseMotionCommand:
  command = env.command_manager.get_term(command_name)
  if isinstance(command, HandBaseMotionCommand):
    return command
  raise TypeError(
    f"Command '{command_name}' is not a HandBaseMotionCommand. Got: {type(command)}"
  )


def _matrix_from_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
  """Decode a 6D rotation representation into a rotation matrix."""
  c1_raw = rot6d[..., 0:3]
  c2_raw = rot6d[..., 3:6]
  c1 = torch.nn.functional.normalize(c1_raw, dim=-1)
  proj = torch.sum(c1 * c2_raw, dim=-1, keepdim=True)
  c2 = torch.nn.functional.normalize(c2_raw - proj * c1, dim=-1)
  c3 = torch.cross(c1, c2, dim=-1)
  return torch.stack((c1, c2, c3), dim=-1)


def _current_hand_pose_in_anchor_frame(
  command: HandBaseMotionCommand,
) -> tuple[torch.Tensor, torch.Tensor]:
  left_robot_idx = int(command.robot_body_indexes[command.left_hand_body_index].item())
  right_robot_idx = int(
    command.robot_body_indexes[command.right_hand_body_index].item()
  )

  hand_pos_w = torch.stack(
    (
      command.robot.data.body_link_pos_w[:, left_robot_idx],
      command.robot.data.body_link_pos_w[:, right_robot_idx],
    ),
    dim=1,
  )
  hand_quat_w = torch.stack(
    (
      command.robot.data.body_link_quat_w[:, left_robot_idx],
      command.robot.data.body_link_quat_w[:, right_robot_idx],
    ),
    dim=1,
  )

  anchor_pos_w = command.robot_anchor_pos_w[:, None, :].expand(-1, 2, -1)
  anchor_quat_w = command.robot_anchor_quat_w[:, None, :].expand(-1, 2, -1)
  hand_pos_b, hand_quat_b = subtract_frame_transforms(
    anchor_pos_w,
    anchor_quat_w,
    hand_pos_w,
    hand_quat_w,
  )
  return hand_pos_b, matrix_from_quat(hand_quat_b)


def hand_position_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward tracking the desired hand positions from the HandBase command."""
  command = _hand_base_command(env, command_name)
  desired_pos_b = command.command[:, :6].reshape(-1, 2, 3)
  actual_pos_b, _ = _current_hand_pose_in_anchor_frame(command)
  error = torch.sum(torch.square(desired_pos_b - actual_pos_b), dim=-1).mean(dim=-1)
  return torch.exp(-error / std**2)


def hand_orientation_tracking_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward tracking the desired hand orientations with geodesic angle error."""
  command = _hand_base_command(env, command_name)
  desired_rot6d_b = command.command[:, 6:18].reshape(-1, 2, 6)
  desired_rotm_b = _matrix_from_rot6d(desired_rot6d_b.reshape(-1, 6)).reshape(
    -1, 2, 3, 3
  )
  _, actual_rotm_b = _current_hand_pose_in_anchor_frame(command)

  relative_rotm = desired_rotm_b.transpose(-2, -1) @ actual_rotm_b
  trace = torch.diagonal(relative_rotm, dim1=-2, dim2=-1).sum(dim=-1)
  cos_angle = torch.clamp((trace - 1.0) * 0.5, min=-1.0 + 1e-6, max=1.0 - 1e-6)
  angle_error = torch.acos(cos_angle)
  squared_geodesic_error = torch.square(angle_error).mean(dim=-1)
  return torch.exp(-squared_geodesic_error / std**2)


def track_base_linear_velocity_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward tracking only the HandBase commanded planar base linear velocity."""
  command = _hand_base_command(env, command_name)
  desired_xy = command.command[:, 18:20]
  actual_lin_vel_b = quat_apply_inverse(
    command.robot_anchor_quat_w,
    command.robot_anchor_lin_vel_w,
  )
  xy_error = torch.sum(torch.square(desired_xy - actual_lin_vel_b[:, :2]), dim=1)
  return torch.exp(-xy_error / std**2)


def track_base_angular_velocity_exp(
  env: ManagerBasedRlEnv,
  command_name: str,
  std: float,
) -> torch.Tensor:
  """Reward tracking only the HandBase commanded base yaw rate."""
  command = _hand_base_command(env, command_name)
  desired_wz = command.command[:, 20]
  actual_ang_vel_b = quat_apply_inverse(
    command.robot_anchor_quat_w,
    command.robot_anchor_ang_vel_w,
  )
  z_error = torch.square(desired_wz - actual_ang_vel_b[:, 2])
  return torch.exp(-z_error / std**2)


def self_collision_cost(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  """Cost that returns the number of self-collisions detected by a sensor."""
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.found is not None
  return sensor.data.found.squeeze(-1)


def feet_contact_force_excess(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  max_contact_force: float,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  force_z = sensor.data.force[..., 2]
  excess = torch.norm(force_z, dim=-1)
  excess = torch.where(
    excess < max_contact_force,
    torch.zeros_like(excess),
    excess - max_contact_force,
  )
  return excess


def feet_slip(
  env: ManagerBasedRlEnv,
  sensor_name: str,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  assert sensor.data.force is not None
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset_cfg.body_ids
  # mjlab's net foot-ground force is negative in world z under load, so use the
  # vertical magnitude as the contact gate.
  contact = torch.abs(sensor.data.force[..., 2]) > 5.0
  foot_speed = torch.norm(asset.data.body_link_lin_vel_w[:, body_ids, :2], dim=-1)
  slip = torch.sqrt(foot_speed) * contact.float()
  return torch.sum(slip, dim=-1)
