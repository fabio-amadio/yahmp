"""Future-stacked joint-position command definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import MotionCommand, MotionCommandCfg
from .representations import (
  future_joint_pos_anchor_rp_representation,
  future_joint_state_anchor_rp_representation,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class FutureJointPosAnchorRpMotionCommand(MotionCommand):
  """Future joint positions with anchor motion terms and roll/pitch."""

  cfg: FutureJointPosAnchorRpMotionCommandCfg

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name != "default":
      return super().get_command_representation(representation_name)
    frames = self.query_motion_frames(self.cfg.command_step_offsets)
    command = future_joint_pos_anchor_rp_representation(frames)
    return command.reshape(self.num_envs, -1)

  @property
  def future_sampling_step_offsets(self) -> tuple[int, ...]:
    return self.cfg.command_step_offsets


@dataclass(kw_only=True)
class FutureJointPosAnchorRpMotionCommandCfg(MotionCommandCfg):
  """Future joint-pos+anchor-motion command with reference roll/pitch."""

  command_step_offsets: tuple[int, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if len(self.command_step_offsets) == 0:
      raise ValueError(
        "`command_step_offsets` must be non-empty for "
        "FutureJointPosAnchorRpMotionCommandCfg."
      )
    return FutureJointPosAnchorRpMotionCommand(self, env)


class FutureJointStateAnchorRpMotionCommand(MotionCommand):
  """Future joint positions and velocities with anchor motion terms."""

  cfg: FutureJointStateAnchorRpMotionCommandCfg

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name != "default":
      return super().get_command_representation(representation_name)
    frames = self.query_motion_frames(self.cfg.command_step_offsets)
    command = future_joint_state_anchor_rp_representation(frames)
    return command.reshape(self.num_envs, -1)

  @property
  def future_sampling_step_offsets(self) -> tuple[int, ...]:
    return self.cfg.command_step_offsets


@dataclass(kw_only=True)
class FutureJointStateAnchorRpMotionCommandCfg(MotionCommandCfg):
  """Future joint-state+anchor-motion command with reference roll/pitch."""

  command_step_offsets: tuple[int, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if len(self.command_step_offsets) == 0:
      raise ValueError(
        "`command_step_offsets` must be non-empty for "
        "FutureJointStateAnchorRpMotionCommandCfg."
      )
    return FutureJointStateAnchorRpMotionCommand(self, env)
