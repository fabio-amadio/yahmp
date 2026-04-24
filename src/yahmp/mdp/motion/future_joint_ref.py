"""Future-stacked joint-reference command definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import MotionCommand, MotionCommandCfg
from .representations import future_joint_ref_anchor_rp_representation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class FutureJointRefAnchorRpMotionCommand(MotionCommand):
  """Future joint references with anchor motion terms and roll/pitch."""

  cfg: FutureJointRefAnchorRpMotionCommandCfg

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name != "default":
      return super().get_command_representation(representation_name)
    frames = self.query_motion_frames(self.cfg.command_step_offsets)
    command = future_joint_ref_anchor_rp_representation(frames)
    return command.reshape(self.num_envs, -1)

  @property
  def future_sampling_step_offsets(self) -> tuple[int, ...]:
    return self.cfg.command_step_offsets


@dataclass(kw_only=True)
class FutureJointRefAnchorRpMotionCommandCfg(MotionCommandCfg):
  """Future joint-ref+anchor-motion command with reference roll/pitch."""

  command_step_offsets: tuple[int, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if len(self.command_step_offsets) == 0:
      raise ValueError(
        "`command_step_offsets` must be non-empty for "
        "FutureJointRefAnchorRpMotionCommandCfg."
      )
    return FutureJointRefAnchorRpMotionCommand(self, env)
