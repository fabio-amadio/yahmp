from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import euler_xyz_from_quat, quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

_DEFAULT_ASSET_CFG = SceneEntityCfg("robot")

__all__ = [
    "PushGoalCommand",
    "PushGoalCommandCfg",
    "push_command",
]


class PushGoalCommand(CommandTerm):
    """Single 2D ground goal; resamples the instant the base reaches it."""

    cfg: PushGoalCommandCfg

    def __init__(self, cfg: PushGoalCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]
        self.crate: Entity = env.scene["crate"]
        self._command = torch.zeros(self.num_envs, 5, device=self.device)
        self.target_pos_w = torch.zeros(self.num_envs, 2, device=self.device)
        self.distance = torch.zeros(self.num_envs, device=self.device)
        self.prev_distance = torch.zeros(self.num_envs, device=self.device)
        self.progress = torch.zeros(self.num_envs, device=self.device)
        self.metrics["distance"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _refresh_command(self) -> None:
        """Goal in the heading frame, that is the robot body frame: rotate (target - root) by -yaw.
        In a nutshell, we are computing the new robot command, in this case a 5D vector, in the body frame of the robot.
        """
        # pos_box_w = self.crate.data.root_com_pos_w  # (N, 3)
        # self.distance = torch.linalg.norm(pos_box_w[:, :2] - self.target_pos_w, dim=-1)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """Resample a new goal for the given envs in the world frame.
        Orientation is indeed the robot yaw w.r.t the world frame, while the position is
        the crate position in the world frame plus a random offset in the range of distance_range and angle_range.
        """
        n = len(env_ids)
        dist = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
        azim = torch.empty(n, device=self.device).uniform_(*self.cfg.angle_range)

        # robot_quat_w = self.robot.data.root_com_quat_w[env_ids]  # (N, 4)
        # _, _, yaw = euler_xyz_from_quat(robot_quat_w)  # (N,) Metodo alternativo per estrarre lo yaw ma heading_w è già disponibile
        yaw = self.robot.data.heading_w[env_ids]  # (N,)
        direction = yaw + azim
        pos_box = self.crate.data.root_com_pos_w[env_ids]  # (N, 3)
        self.target_pos_w[env_ids, 0] = pos_box[:, 0] + dist * torch.cos(direction)
        self.target_pos_w[env_ids, 1] = pos_box[:, 1] + dist * torch.sin(direction)

        new_dist = torch.linalg.norm(
            pos_box[:, :2] - self.target_pos_w[env_ids], dim=-1
        )
        self.distance[env_ids] = new_dist
        self.prev_distance[env_ids] = new_dist
        self.progress[env_ids] = 0.0

    def _update_command(self) -> None:
        ## Distance between box and target position in world frame
        pos_box_w = self.crate.data.root_com_pos_w  # (N, 3)
        self.distance = torch.linalg.norm(pos_box_w[:, :2] - self.target_pos_w, dim=-1)
        self.progress = self.prev_distance - self.distance
        self.prev_distance = self.distance.clone()

        ## Compute the command in the robot body frame
        yaw = self.robot.data.heading_w  # (N,)
        pos_robot_w = self.robot.data.root_com_pos_w  # (N, 3)
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        dx = pos_box_w[:, 0] - pos_robot_w[:, 0]
        dy = pos_box_w[:, 1] - pos_robot_w[:, 1]
        self._command[:, 0] = (
            cos_y * dx + sin_y * dy
        )  # x_b ##è una rotazione attorno a Z trasposta cosìcché world frame to body frame
        self._command[:, 1] = -sin_y * dx + cos_y * dy

        gx = self.target_pos_w[:, 0] - pos_box_w[:, 0]
        gy = self.target_pos_w[:, 1] - pos_box_w[:, 1]
        self._command[:, 2] = cos_y * gx + sin_y * gy
        self._command[:, 3] = (
            -sin_y * gx + cos_y * gy
        )  ##Lo definiamo sempre nell'orientamento del robot, ci interessa dove punta il robot no la scatola
        self._command[:, 4] = self.distance

        ##Oppure con i quaternioni:
        # pos_robot_w = self.robot.data.root_com_pos_w  # (N, 3)
        # yaw_q = yaw_quat(self.robot.data.root_com_quat_w)  # (N, 4)

        # delta_crate = pos_box_w - pos_robot_w  # (N, 3)
        # delta_target = torch.zeros_like(delta_crate)
        # delta_target[:, :2] = self.target_pos_w - pos_box_w[:, :2]
        # delta = pos_box_w[:, :2] - self.target_pos_w

    def _update_metrics(self) -> None:
        self.metrics["distance"] = self.distance

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        pass


@dataclass(kw_only=True)
class PushGoalCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    resampling_time_range: tuple[float, float] = (
        1e9,
        1e9,
    )
    distance_range: tuple[float, float] = (0.5, 1.5)
    angle_range: tuple[float, float] = (-0.5, 0.5)

    def build(self, env: ManagerBasedRlEnv) -> PushGoalCommand:
        return PushGoalCommand(self, env)


def push_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return the heading-frame goal command ``[x_b, y_b, dist]``."""
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return command


def _push_command(env: ManagerBasedRlEnv, command_name: str) -> PushGoalCommand:
    command = env.command_manager.get_term(command_name)
    if isinstance(command, PushGoalCommand):
        return command
    raise TypeError(
        f"Command '{command_name}' is not a PushGoalCommand. Got: {type(command)}"
    )
