"""Single-step joint-position motion command definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import MotionCommand, MotionCommandCfg
from .representations import (
  joint_pos_anchor_rp_representation,
  joint_state_anchor_rp_representation,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class JointPosAnchorRpMotionCommand(MotionCommand):
  """Single-step joint positions with anchor motion terms and roll/pitch."""

  cfg: JointPosAnchorRpMotionCommandCfg

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name == "default":
      return joint_pos_anchor_rp_representation(
        self.joint_pos,
        self.anchor_pos_w,
        self.anchor_quat_w,
        self.anchor_lin_vel_w,
        self.anchor_ang_vel_w,
      )
    return super().get_command_representation(representation_name)


@dataclass(kw_only=True)
class JointPosAnchorRpMotionCommandCfg(MotionCommandCfg):
  """Configuration for single-step joint-pos+anchor roll/pitch commands."""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return JointPosAnchorRpMotionCommand(self, env)


class JointStateAnchorRpMotionCommand(MotionCommand):
  """Single-step joint positions and velocities with anchor motion terms."""

  cfg: JointStateAnchorRpMotionCommandCfg

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name == "default":
      return joint_state_anchor_rp_representation(
        self.joint_pos,
        self.joint_vel,
        self.anchor_pos_w,
        self.anchor_quat_w,
        self.anchor_lin_vel_w,
        self.anchor_ang_vel_w,
      )
    return super().get_command_representation(representation_name)


@dataclass(kw_only=True)
class JointStateAnchorRpMotionCommandCfg(MotionCommandCfg):
  """Configuration for single-step joint-state+anchor roll/pitch commands."""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return JointStateAnchorRpMotionCommand(self, env)
