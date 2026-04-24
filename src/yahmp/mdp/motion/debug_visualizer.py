"""Debug visualization helpers for YAHMP motion commands."""

from __future__ import annotations

import copy
from typing import TYPE_CHECKING

import numpy as np
from mjlab.viewer.debug_visualizer import DebugVisualizer

if TYPE_CHECKING:
  from .base import MotionCommand


def debug_visualize_motion_command(
  command: MotionCommand,
  visualizer: DebugVisualizer,
) -> None:
  """Render the current reference state as a ghost articulation."""
  env_indices = visualizer.get_env_indices(command.num_envs)
  if not env_indices:
    return

  if command._ghost_model is None:
    command._ghost_model = copy.deepcopy(command._env.sim.mj_model)
    command._ghost_model.geom_rgba[:] = command._ghost_color

  entity = command._env.scene[command.cfg.entity_name]
  indexing = entity.indexing
  free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
  joint_q_adr = indexing.joint_q_adr.cpu().numpy()

  for batch in env_indices:
    qpos = np.zeros(command._env.sim.mj_model.nq)
    # Free-joint pose must use the articulated root body (not tracking anchor).
    qpos[free_joint_q_adr[0:3]] = command.root_pos_w[batch].cpu().numpy()
    qpos[free_joint_q_adr[3:7]] = command.root_quat_w[batch].cpu().numpy()
    qpos[joint_q_adr] = command.joint_pos[batch].cpu().numpy()
    visualizer.add_ghost_mesh(qpos, model=command._ghost_model, label=f"ghost_{batch}")
