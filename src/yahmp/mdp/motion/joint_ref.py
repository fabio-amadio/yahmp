"""Single-step joint-reference motion command definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import MotionCommand, MotionCommandCfg
from .representations import joint_ref_anchor_rp_representation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class JointRefAnchorRpMotionCommand(MotionCommand):
  """Single-step joint references with anchor motion terms and roll/pitch."""

  cfg: JointRefAnchorRpMotionCommandCfg

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name == "default":
      return joint_ref_anchor_rp_representation(
        self.joint_pos,
        self.joint_vel,
        self.anchor_pos_w,
        self.anchor_quat_w,
        self.anchor_lin_vel_w,
        self.anchor_ang_vel_w,
      )
    return super().get_command_representation(representation_name)


@dataclass(kw_only=True)
class JointRefAnchorRpMotionCommandCfg(MotionCommandCfg):
  """Configuration for single-step joint-ref+anchor roll/pitch commands."""

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    return JointRefAnchorRpMotionCommand(self, env)
