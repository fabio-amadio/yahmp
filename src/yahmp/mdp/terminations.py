from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from .motion.base import MotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def motion_ref_expired(
  env: ManagerBasedRlEnv,
  command_name: str,
) -> torch.Tensor:
  """Terminate when the current clip would step past its final reference frame."""
  command = cast(MotionCommand, env.command_manager.get_term(command_name))
  return command.motion_expired(step_offset=1)
