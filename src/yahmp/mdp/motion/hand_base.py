"""Single-step hand-pose + base-velocity/height student command definitions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_apply_inverse,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .base import MotionCommand, MotionCommandCfg
from .debug_visualizer import debug_visualize_motion_command
from .representations import hand_base_representation

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


def _matrix_from_rot6d(rot6d: torch.Tensor) -> torch.Tensor:
  """Decode a 6D rotation representation into a rotation matrix."""
  c1_raw = rot6d[..., 0:3]
  c2_raw = rot6d[..., 3:6]
  c1 = torch.nn.functional.normalize(c1_raw, dim=-1)
  proj = torch.sum(c1 * c2_raw, dim=-1, keepdim=True)
  c2 = torch.nn.functional.normalize(c2_raw - proj * c1, dim=-1)
  c3 = torch.cross(c1, c2, dim=-1)
  return torch.stack((c1, c2, c3), dim=-1)


class HandBaseMotionCommand(MotionCommand):
  """Current hand poses plus planar base velocity and anchor height."""

  cfg: HandBaseMotionCommandCfg

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)
    assert isinstance(cfg, HandBaseMotionCommandCfg)
    self.cfg = cfg

    name_to_idx = {name: idx for idx, name in enumerate(self.cfg.body_names)}
    try:
      self._left_hand_idx = name_to_idx[self.cfg.left_hand_body_name]
      self._right_hand_idx = name_to_idx[self.cfg.right_hand_body_name]
    except KeyError as exc:
      raise ValueError(
        "Hand body names must be included in `body_names` for HandBaseMotionCommand. "
        f"Missing: {exc}"
      ) from exc

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    if representation_name == "default":
      return hand_base_representation(
        body_pos_w=self.body_pos_w,
        body_quat_w=self.body_quat_w,
        anchor_pos_w=self.anchor_pos_w,
        anchor_quat_w=self.anchor_quat_w,
        anchor_lin_vel_w=self.anchor_lin_vel_w,
        anchor_ang_vel_w=self.anchor_ang_vel_w,
        left_hand_body_index=self._left_hand_idx,
        right_hand_body_index=self._right_hand_idx,
      )
    return super().get_command_representation(representation_name)

  @property
  def left_hand_body_index(self) -> int:
    return self._left_hand_idx

  @property
  def right_hand_body_index(self) -> int:
    return self._right_hand_idx

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    if self.cfg.show_ghost:
      debug_visualize_motion_command(self, visualizer)

    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    left_robot_idx = int(self.robot_body_indexes[self._left_hand_idx].item())
    right_robot_idx = int(self.robot_body_indexes[self._right_hand_idx].item())

    current_hand_pos_w = torch.stack(
      (
        self.robot.data.body_link_pos_w[:, left_robot_idx],
        self.robot.data.body_link_pos_w[:, right_robot_idx],
      ),
      dim=1,
    )
    current_hand_quat_w = torch.stack(
      (
        self.robot.data.body_link_quat_w[:, left_robot_idx],
        self.robot.data.body_link_quat_w[:, right_robot_idx],
      ),
      dim=1,
    )
    current_hand_rotm_w = matrix_from_quat(current_hand_quat_w)

    command = self.command
    desired_hand_pos_b = command[:, :6].reshape(self.num_envs, 2, 3)
    desired_hand_rot6d_b = command[:, 6:18].reshape(self.num_envs, 2, 6)
    desired_hand_rotm_b = _matrix_from_rot6d(
      desired_hand_rot6d_b.reshape(self.num_envs * 2, 6)
    ).reshape(self.num_envs, 2, 3, 3)

    anchor_pos_w = self.robot_anchor_pos_w
    anchor_quat_w = self.robot_anchor_quat_w
    anchor_rotm_w = matrix_from_quat(anchor_quat_w)
    anchor_quat_w_hands = anchor_quat_w[:, None, :].expand(-1, 2, -1)
    desired_hand_pos_w = anchor_pos_w[:, None, :] + quat_apply(
      anchor_quat_w_hands, desired_hand_pos_b
    )
    desired_hand_rotm_w = anchor_rotm_w[:, None, :, :] @ desired_hand_rotm_b

    actual_lin_vel_b = quat_apply_inverse(
      self.robot_anchor_quat_w, self.robot_anchor_lin_vel_w
    )
    actual_ang_vel_b = quat_apply_inverse(
      self.robot_anchor_quat_w, self.robot_anchor_ang_vel_w
    )

    arrow_scale = float(self.cfg.viz_scale)
    z_offset = float(self.cfg.viz_z_offset)
    hand_names = ("left_hand", "right_hand")

    for batch in env_indices:
      for hand_i, hand_name in enumerate(hand_names):
        visualizer.add_frame(
          position=desired_hand_pos_w[batch, hand_i].cpu().numpy(),
          rotation_matrix=desired_hand_rotm_w[batch, hand_i].cpu().numpy(),
          scale=0.12,
          label=f"desired_{hand_name}_{batch}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )
        visualizer.add_frame(
          position=current_hand_pos_w[batch, hand_i].cpu().numpy(),
          rotation_matrix=current_hand_rotm_w[batch, hand_i].cpu().numpy(),
          scale=0.14,
          label=f"current_{hand_name}_{batch}",
        )

      base_pos_w = anchor_pos_w[batch].cpu().numpy()
      base_rotm_w = anchor_rotm_w[batch].cpu().numpy()
      if np.linalg.norm(base_pos_w) < 1e-6:
        continue

      cmd_vx = float(command[batch, 18].item())
      cmd_vy = float(command[batch, 19].item())
      cmd_wz = float(command[batch, 20].item())
      act_vx = float(actual_lin_vel_b[batch, 0].item())
      act_vy = float(actual_lin_vel_b[batch, 1].item())
      act_wz = float(actual_ang_vel_b[batch, 2].item())

      def local_to_world(
        local_vec: np.ndarray,
        *,
        base_pos_w: np.ndarray = base_pos_w,
        base_rotm_w: np.ndarray = base_rotm_w,
      ) -> np.ndarray:
        return base_pos_w + base_rotm_w @ local_vec

      arrow_origin = local_to_world(np.array([0.0, 0.0, z_offset]) * arrow_scale)

      visualizer.add_arrow(
        arrow_origin,
        local_to_world(
          (np.array([0.0, 0.0, z_offset]) + np.array([cmd_vx, cmd_vy, 0.0]))
          * arrow_scale
        ),
        color=(0.2, 0.2, 0.6, 0.6),
        width=0.015,
        label=f"cmd_lin_{batch}",
      )
      visualizer.add_arrow(
        arrow_origin,
        local_to_world(
          (np.array([0.0, 0.0, z_offset]) + np.array([0.0, 0.0, cmd_wz])) * arrow_scale
        ),
        color=(0.2, 0.6, 0.2, 0.6),
        width=0.015,
        label=f"cmd_ang_{batch}",
      )
      visualizer.add_arrow(
        arrow_origin,
        local_to_world(
          (np.array([0.0, 0.0, z_offset]) + np.array([act_vx, act_vy, 0.0]))
          * arrow_scale
        ),
        color=(0.0, 0.6, 1.0, 0.7),
        width=0.015,
        label=f"act_lin_{batch}",
      )
      visualizer.add_arrow(
        arrow_origin,
        local_to_world(
          (np.array([0.0, 0.0, z_offset]) + np.array([0.0, 0.0, act_wz])) * arrow_scale
        ),
        color=(0.0, 1.0, 0.4, 0.7),
        width=0.015,
        label=f"act_ang_{batch}",
      )


@dataclass(kw_only=True)
class HandBaseMotionCommandCfg(MotionCommandCfg):
  """Configuration for hand-pose + base-velocity student commands."""

  left_hand_body_name: str = ""
  right_hand_body_name: str = ""
  show_ghost: bool = True
  viz_scale: float = 0.5
  viz_z_offset: float = 0.2

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    if self.left_hand_body_name == "" or self.right_hand_body_name == "":
      raise ValueError(
        "`left_hand_body_name` and `right_hand_body_name` must be set for "
        "HandBaseMotionCommandCfg."
      )
    return HandBaseMotionCommand(self, env)
