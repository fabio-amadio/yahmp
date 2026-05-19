"""Synchronized student/teacher joint-position commands for distillation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from .base import MotionCommand
from .joint_ref import JointPosAnchorRpMotionCommand, JointPosAnchorRpMotionCommandCfg
from .representations import (
  future_joint_pos_anchor_rp_representation,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class TeacherStudentJointPosAnchorRpMotionCommand(JointPosAnchorRpMotionCommand):
  """Joint-pos+anchor+roll/pitch student command with synchronized teacher view."""

  cfg: TeacherStudentJointPosAnchorRpMotionCommandCfg

  @property
  def command_representation_names(self) -> tuple[str, ...]:
    return ("default", "teacher")

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name == "teacher":
      frames = self.query_motion_frames(self.cfg.future_sampling_step_offsets)
      return future_joint_pos_anchor_rp_representation(frames).reshape(
        self.num_envs, -1
      )
    return super().get_command_representation(representation_name)

  @property
  def future_sampling_step_offsets(self) -> tuple[int, ...]:
    return self.cfg.future_sampling_step_offsets


@dataclass(kw_only=True)
class TeacherStudentJointPosAnchorRpMotionCommandCfg(JointPosAnchorRpMotionCommandCfg):
  """Configuration for synchronized YAHMP-style student and teacher command views."""

  future_sampling_step_offsets: tuple[int, ...] = ()

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if len(self.future_sampling_step_offsets) == 0:
      raise ValueError(
        "`future_sampling_step_offsets` must be non-empty for "
        "TeacherStudentJointPosAnchorRpMotionCommandCfg."
      )
    return TeacherStudentJointPosAnchorRpMotionCommand(self, env)
