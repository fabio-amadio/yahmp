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
    "push_progress",
    "push_success_bonus",
    "push_velocity_tracking_exp",
    "push_position_tracking_exp",
    "push_hands_on_crate",
    "push_command_levels",
    "reached_termination",
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
        self.just_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _refresh_command(self) -> None:
        """Goal in the heading frame, that is the robot body frame: rotate (target - root) by -yaw.
        In a nutshell, we are computing the new robot command, in this case a 5D vector, in the body frame of the robot.
        """
        pass
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

        # Crate reset position (world), DETERMINISTIC: default_root_state (=
        # init_state, env-local) + env_origins. We must NOT read
        # ``crate.data.root_com_pos_w`` here: this runs inside the reset event,
        # where derived quantities are still STALE (the drifted end-of-episode
        # crate pose), which would anchor the goal to the wrong spot. The crate
        # is re-placed to exactly this pose by the ``reset_crate`` event.
        origins = self._env.scene.env_origins[env_ids, :2]
        crate_xy = self.crate.data.default_root_state[env_ids, :2] + origins

        # Goal in front of the robot: after the constrained base reset the robot
        # faces ~+x (world), so sample the goal in the +x direction (+ azim)
        # beyond the crate. (The robot's fresh yaw is likewise unavailable here,
        # so we use the nominal forward instead of the stale ``heading_w``.)
        direction = azim
        self.target_pos_w[env_ids, 0] = crate_xy[:, 0] + dist * torch.cos(direction)
        self.target_pos_w[env_ids, 1] = crate_xy[:, 1] + dist * torch.sin(direction)

        new_dist = torch.linalg.norm(crate_xy - self.target_pos_w[env_ids], dim=-1)
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

        self.just_reached = self.distance < self.cfg.reach_tol

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
    reach_tol: float = 0.2

    def build(self, env: ManagerBasedRlEnv) -> PushGoalCommand:
        return PushGoalCommand(self, env)


def push_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return the 5D command ``[crate_x_b, crate_y_b, goal_dx_b, goal_dy_b, dist]``
    (crate relative to robot + goal relative to crate, both in the robot heading
    frame, plus the crate->goal distance)."""
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return command


def push_progress(env: ManagerBasedRlEnv, command_name: str, cap=2.0) -> torch.Tensor:
    """Return the progress made towards the goal command. Linear reward for reaching the goal."""
    command = _push_command(env, command_name)
    return torch.clamp(command.progress / max(env.step_dt, 1e-6), -cap, cap)


def push_success_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return a bonus for reaching the goal command."""
    command = _push_command(env, command_name)
    return command.just_reached.float()


def push_velocity_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, cruise_speed: float = 1.1
) -> torch.Tensor:
    ## Push the robot to walk rather than trying to maximize the return always running
    command = _push_command(env, command_name)
    speed = torch.linalg.norm(
        command.robot.data.root_com_lin_vel_w[..., :2], dim=-1
    )  # (N,)
    return torch.exp(-torch.square(speed - cruise_speed) / (std**2))


def push_position_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    command = _push_command(env, command_name)
    return torch.exp(-torch.square(command.distance) / (std**2))


def reached_termination(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return a boolean tensor indicating whether the goal has been reached."""
    command = _push_command(env, command_name)
    return command.distance < command.cfg.reach_tol


def push_hands_on_crate(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float = 0.4,
    asset_cfg=SceneEntityCfg("robot", body_names=(".*_wrist_yaw_link",)),
) -> torch.Tensor:
    command = _push_command(env, command_name)
    hands = command.robot.data.body_link_pos_w[:, asset_cfg.body_ids]  # (N,2,3)
    crate = env.scene["crate"].data.root_link_pos_w
    d = torch.linalg.norm(hands - crate.unsqueeze(1), dim=-1)
    return torch.exp(-torch.square(d) / (std**2)).mean(dim=-1)


def _push_command(env: ManagerBasedRlEnv, command_name: str) -> PushGoalCommand:
    command = env.command_manager.get_term(command_name)
    if isinstance(command, PushGoalCommand):
        return command
    raise TypeError(
        f"Command '{command_name}' is not a PushGoalCommand. Got: {type(command)}"
    )


class PushLevelStage(TypedDict, total=False):
    step: int
    distance_range: tuple[float, float]
    angle_range: tuple[float, float]


def push_command_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    command_name: str,
    stages: list[PushLevelStage],
) -> dict[str, float]:
    """Curriculum: widen the goal spawn distance and azimuth over training."""
    del env_ids
    command = cast(PushGoalCommand, env.command_manager.get_term(command_name))
    distance_range = command.cfg.distance_range
    angle_range = command.cfg.angle_range
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            distance_range = stage["distance_range"]
            angle_range = stage["angle_range"]
    command.cfg.distance_range = distance_range
    command.cfg.angle_range = angle_range
    return {
        "distance_max": float(distance_range[1]),
        "angle_max": float(angle_range[1]),
    }
