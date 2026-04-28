from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.envs import mdp as builtin_mdp
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactSensor
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply_inverse,
  subtract_frame_transforms,
)

from .motion.base import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")


def _rot6d_from_matrix(rotm: torch.Tensor) -> torch.Tensor:
  """Encode rotation matrix as [c1(3), c2(3)] (first two columns)."""
  first_two_cols = rotm[..., :, :2]
  return first_two_cols.transpose(-2, -1).reshape(rotm.shape[0], -1)


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


def _yaw_only_quat(quat_wxyz: torch.Tensor) -> torch.Tensor:
  _, _, yaw = _quat_roll_pitch_yaw(quat_wxyz)
  half_yaw = 0.5 * yaw
  zeros = torch.zeros_like(half_yaw)
  return torch.stack((torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1)


def _expand_model_field(
  current: torch.Tensor, default: torch.Tensor, num_envs: int
) -> torch.Tensor:
  if current.ndim == default.ndim:
    return current.unsqueeze(0).expand(num_envs, *current.shape)
  return current


def _resolve_command(env: ManagerBasedRlEnv, command_name: str) -> MotionCommand:
  return cast(MotionCommand, env.command_manager.get_term(command_name))


def _body_positions_local(
  anchor_pos_w: torch.Tensor,
  anchor_quat_w: torch.Tensor,
  body_pos_w: torch.Tensor,
) -> torch.Tensor:
  num_envs, num_bodies = body_pos_w.shape[:2]
  offsets_w = body_pos_w - anchor_pos_w[:, None, :]
  local_pos = quat_apply_inverse(
    anchor_quat_w[:, None, :].expand(-1, num_bodies, -1).reshape(-1, 4),
    offsets_w.reshape(-1, 3),
  )
  return local_pos.reshape(num_envs, num_bodies, 3)


def _key_body_positions_local(
  command: MotionCommand, yaw_only: bool = False, reference: bool = False
) -> torch.Tensor:
  if reference:
    anchor_pos_w = command.anchor_pos_w
    anchor_quat_w = command.anchor_quat_w
    body_pos_w = command.body_pos_w
  else:
    anchor_pos_w = command.robot_anchor_pos_w
    anchor_quat_w = command.robot_anchor_quat_w
    body_pos_w = command.robot_body_pos_w

  if yaw_only:
    anchor_quat_w = _yaw_only_quat(anchor_quat_w)
  return _body_positions_local(anchor_pos_w, anchor_quat_w, body_pos_w)


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _resolve_command(env, command_name)

  pos, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _resolve_command(env, command_name)

  _, ori = subtract_frame_transforms(
    command.robot_anchor_pos_w,
    command.robot_anchor_quat_w,
    command.anchor_pos_w,
    command.anchor_quat_w,
  )
  return _rot6d_from_matrix(matrix_from_quat(ori))


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _resolve_command(env, command_name)

  num_bodies = len(command.cfg.body_names)
  pos_b, _ = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = _resolve_command(env, command_name)

  num_bodies = len(command.cfg.body_names)
  _, ori_b = subtract_frame_transforms(
    command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
    command.robot_body_pos_w,
    command.robot_body_quat_w,
  )
  return _rot6d_from_matrix(matrix_from_quat(ori_b))


def feet_contact_mask(env: ManagerBasedRlEnv, sensor_name: str) -> torch.Tensor:
  sensor: ContactSensor = env.scene[sensor_name]
  sensor_data = sensor.data
  assert sensor_data.found is not None
  return (sensor_data.found > 0).float()


def motion_mass_params(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  body_ids = asset.indexing.body_ids[asset_cfg.body_ids]
  if body_ids.numel() != 1:
    raise ValueError("motion_mass_params expects exactly one root body.")
  body_id = int(body_ids.reshape(-1)[0].item())

  current_body_mass = _expand_model_field(
    env.sim.model.body_mass,
    env.sim.get_default_field("body_mass"),
    env.num_envs,
  )
  current_body_ipos = _expand_model_field(
    env.sim.model.body_ipos,
    env.sim.get_default_field("body_ipos"),
    env.num_envs,
  )
  default_body_mass = env.sim.get_default_field("body_mass")[body_id]
  default_body_ipos = env.sim.get_default_field("body_ipos")[body_id]

  mass_delta = current_body_mass[:, body_id : body_id + 1] - default_body_mass
  com_delta = current_body_ipos[:, body_id, :] - default_body_ipos
  return torch.cat((mass_delta, com_delta), dim=-1)


def motion_friction_coeff(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  geom_ids = asset.indexing.geom_ids[asset_cfg.geom_ids]
  current_friction = _expand_model_field(
    env.sim.model.geom_friction,
    env.sim.get_default_field("geom_friction"),
    env.num_envs,
  )
  return current_friction[:, geom_ids, 0].mean(dim=-1, keepdim=True)


def motor_strength_kp_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  ctrl_ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]

  current_gainprm = _expand_model_field(
    env.sim.model.actuator_gainprm,
    env.sim.get_default_field("actuator_gainprm"),
    env.num_envs,
  )
  default_gainprm = env.sim.get_default_field("actuator_gainprm")
  current_kp = current_gainprm[:, ctrl_ids, 0]
  default_kp = default_gainprm[ctrl_ids, 0].abs().clamp_min(1.0e-6)
  return current_kp.abs() / default_kp - 1.0


def motor_strength_kd_rel(
  env: ManagerBasedRlEnv,
  asset_cfg: SceneEntityCfg = _DEFAULT_ASSET_CFG,
) -> torch.Tensor:
  asset: Entity = env.scene[asset_cfg.name]
  ctrl_ids = asset.indexing.ctrl_ids[asset_cfg.actuator_ids]

  current_biasprm = _expand_model_field(
    env.sim.model.actuator_biasprm,
    env.sim.get_default_field("actuator_biasprm"),
    env.num_envs,
  )
  default_biasprm = env.sim.get_default_field("actuator_biasprm")
  current_kd = current_biasprm[:, ctrl_ids, 2].abs()
  default_kd = default_biasprm[ctrl_ids, 2].abs().clamp_min(1.0e-6)
  return current_kd / default_kd - 1.0


def motion_command_representation(
  env: ManagerBasedRlEnv,
  command_name: str,
  representation_name: str = "default",
) -> torch.Tensor:
  command = _resolve_command(env, command_name)
  return command.get_command_representation(representation_name)


def motion_student_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return motion_command_representation(env, command_name, representation_name="default")


def motion_teacher_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  return motion_command_representation(env, command_name, representation_name="teacher")


def motion_teacher_actor_command(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  return torch.cat(
    (
      motion_student_command(env, command_name),
      motion_teacher_command(env, command_name),
    ),
    dim=-1,
  )


def motion_teacher_policy_command(
  env: ManagerBasedRlEnv, command_name: str
) -> torch.Tensor:
  """Backward-compatible alias for the teacher actor command representation."""
  return motion_teacher_actor_command(env, command_name)


def motion_first_step_command(
  env: ManagerBasedRlEnv,
  command_name: str,
  representation_name: str = "default",
) -> torch.Tensor:
  """Return the first per-step command from a stacked motion representation."""
  command = _resolve_command(env, command_name)
  command_obs = command.get_command_representation(representation_name)
  num_steps = max(len(command.future_sampling_step_offsets), 1)
  if num_steps <= 1 or command_obs.shape[-1] % num_steps != 0:
    return command_obs
  return command_obs.reshape(env.num_envs, num_steps, -1)[:, 0]


def _yahmp_current_observation(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Return YAHMP's deployment-ready current block.

  Layout: first motion-command step followed by proprio
  [base_ang_vel, projected_gravity, joint_pos_rel, joint_vel_rel, last_action].
  """
  return torch.cat(
    (
      motion_first_step_command(env, command_name=command_name),
      builtin_mdp.builtin_sensor(env, sensor_name="robot/imu_ang_vel"),
      builtin_mdp.projected_gravity(env),
      builtin_mdp.joint_pos_rel(env),
      builtin_mdp.joint_vel_rel(env),
      builtin_mdp.last_action(env),
    ),
    dim=-1,
  )


def _yahmp_current_observation_with_privileged(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Return YAHMP's current block augmented with privileged observations."""
  return torch.cat(
    (
      _yahmp_current_observation(env, command_name),
      builtin_mdp.builtin_sensor(env, sensor_name="robot/imu_lin_vel"),
      motion_anchor_pos_b(env, command_name),
      motion_anchor_ori_b(env, command_name),
      robot_body_pos_b(env, command_name),
      robot_body_ori_b(env, command_name),
      feet_contact_mask(env, sensor_name="feet_ground_contact"),
      motion_friction_coeff(env, asset_cfg=SceneEntityCfg("robot", geom_names=())),
    ),
    dim=-1,
  )


class YahmpObservationHistory:
  """Time-major history buffer for YAHMP actor observations.

  The buffer stores the deployment-ready current block
  ``[first_step_motion_command, proprio]`` and returns the previous
  ``history_length`` entries flattened as ``[t-H, ..., t-1]``.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self.command_name = cfg.params["command_name"]
    self.history_length = int(cfg.params["history_length"])
    self.include_privileged = bool(cfg.params.get("include_privileged", False))
    if self.history_length <= 0:
      raise ValueError("YahmpObservationHistory requires history_length > 0.")
    self._history: torch.Tensor | None = None
    self._needs_reset = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if self._history is None:
      return
    if env_ids is None or isinstance(env_ids, slice):
      self._needs_reset[:] = True
    else:
      self._needs_reset[env_ids] = True

  def __call__(
    self,
    env: ManagerBasedRlEnv,
    command_name: str | None = None,
    history_length: int | None = None,
    include_privileged: bool | None = None,
  ) -> torch.Tensor:
    del history_length  # Config-time validation keeps this fixed.
    del include_privileged  # Config-time validation keeps this fixed.
    active_command_name = command_name or self.command_name
    if self.include_privileged:
      current = _yahmp_current_observation_with_privileged(env, active_command_name)
    else:
      current = _yahmp_current_observation(env, active_command_name)

    if self._history is None:
      self._history = current[:, None, :].repeat(1, self.history_length, 1)
      self._needs_reset = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
      return self._history.reshape(env.num_envs, -1)

    if torch.any(self._needs_reset):
      reset_ids = torch.nonzero(self._needs_reset, as_tuple=False).squeeze(-1)
      self._history[reset_ids] = current[reset_ids, None, :].repeat(
        1, self.history_length, 1
      )
      self._needs_reset[reset_ids] = False

    history_out = self._history.reshape(env.num_envs, -1).clone()
    self._history[:, :-1] = self._history[:, 1:].clone()
    self._history[:, -1] = current
    return history_out
