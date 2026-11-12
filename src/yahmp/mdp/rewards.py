from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor

from .motion.base import MotionCommand

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
