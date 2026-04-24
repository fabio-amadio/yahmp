from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.envs.mdp.actions.actions import BaseAction, JointPositionActionCfg

from .motion.base import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class ResidualJointPositionActionCfg(JointPositionActionCfg):
  """Joint position action interpreted as a residual over motion reference."""

  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> ResidualJointPositionAction:
    return ResidualJointPositionAction(self, env)


class ResidualJointPositionAction(BaseAction):
  """Track motion joint targets with additive policy residuals."""

  cfg: ResidualJointPositionActionCfg

  def __init__(self, cfg: ResidualJointPositionActionCfg, env: ManagerBasedRlEnv):
    if cfg.use_default_offset:
      raise ValueError(
        "ResidualJointPositionActionCfg requires use_default_offset=False because "
        "the motion command provides the position reference."
      )
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    command = cast(
      MotionCommand,
      self._env.command_manager.get_term(self.cfg.command_name),
    )
    reference = command.joint_pos[:, self._target_ids]
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    target = reference + self._processed_actions - encoder_bias
    self._entity.set_joint_position_target(target, joint_ids=self._target_ids)


@dataclass(kw_only=True)
class JointPositionVelocityActionCfg(JointPositionActionCfg):
  """Classic joint position action augmented with a filtered velocity reference."""

  vel_ref_alpha: float = 0.5
  vel_ref_eta: float = 1.0
  vel_ref_dt: float | None = None

  def build(self, env: ManagerBasedRlEnv) -> JointPositionVelocityAction:
    return JointPositionVelocityAction(self, env)


@dataclass(kw_only=True)
class ResidualJointPositionVelocityActionCfg(JointPositionVelocityActionCfg):
  """Residual joint position action augmented with a filtered velocity reference."""

  command_name: str = "motion"

  def build(self, env: ManagerBasedRlEnv) -> ResidualJointPositionVelocityAction:
    return ResidualJointPositionVelocityAction(self, env)


class JointPositionVelocityAction(BaseAction):
  """Set joint position targets and synthesize a velocity reference from them."""

  cfg: JointPositionVelocityActionCfg

  def __init__(
    self,
    cfg: JointPositionVelocityActionCfg,
    env: ManagerBasedRlEnv,
  ):
    if not 0.0 <= cfg.vel_ref_alpha <= 1.0:
      raise ValueError(f"vel_ref_alpha must lie in [0, 1], got {cfg.vel_ref_alpha}.")
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._entity.data.default_joint_pos[:, self._target_ids].clone()

    self._vel_ref_dt = (
      float(cfg.vel_ref_dt) if cfg.vel_ref_dt is not None else float(self._env.step_dt)
    )
    if self._vel_ref_dt <= 0.0:
      raise ValueError(f"vel_ref_dt must be positive, got {self._vel_ref_dt}.")

    self._prev_position_reference = torch.zeros_like(self._processed_actions)
    self._velocity_reference = torch.zeros_like(self._processed_actions)
    self._has_prev_reference = torch.zeros(
      self.num_envs, dtype=torch.bool, device=self.device
    )

  def _position_reference(self) -> torch.Tensor:
    return self._processed_actions

  def process_actions(self, actions: torch.Tensor):
    super().process_actions(actions)

    position_reference = self._position_reference()
    raw_velocity_reference = (
      position_reference - self._prev_position_reference
    ) / self._vel_ref_dt
    first_step_mask = ~self._has_prev_reference
    raw_velocity_reference[first_step_mask] = 0.0

    alpha = self.cfg.vel_ref_alpha
    self._velocity_reference += alpha * (
      raw_velocity_reference - self._velocity_reference
    )
    self._velocity_reference[first_step_mask] = 0.0

    self._prev_position_reference[:] = position_reference
    self._has_prev_reference[:] = True

  def apply_actions(self) -> None:
    encoder_bias = self._entity.data.encoder_bias[:, self._target_ids]
    position_target = self._position_reference() - encoder_bias
    velocity_target = self.cfg.vel_ref_eta * self._velocity_reference
    self._entity.set_joint_position_target(position_target, joint_ids=self._target_ids)
    self._entity.set_joint_velocity_target(velocity_target, joint_ids=self._target_ids)

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    super().reset(env_ids=env_ids)
    self._prev_position_reference[env_ids] = 0.0
    self._velocity_reference[env_ids] = 0.0
    self._has_prev_reference[env_ids] = False


class ResidualJointPositionVelocityAction(JointPositionVelocityAction):
  """Track motion joint targets with residual position and filtered velocity refs."""

  cfg: ResidualJointPositionVelocityActionCfg

  def __init__(
    self,
    cfg: ResidualJointPositionVelocityActionCfg,
    env: ManagerBasedRlEnv,
  ):
    if cfg.use_default_offset:
      raise ValueError(
        "ResidualJointPositionVelocityActionCfg requires use_default_offset=False "
        "because the motion command provides the position reference."
      )
    super().__init__(cfg=cfg, env=env)

  def _position_reference(self) -> torch.Tensor:
    command = cast(
      MotionCommand,
      self._env.command_manager.get_term(self.cfg.command_name),
    )
    reference = command.joint_pos[:, self._target_ids]
    return reference + self._processed_actions
