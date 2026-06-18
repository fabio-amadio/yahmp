from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, TypedDict, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
    "ReachTargetCommand",
    "ReachTargetCommandCfg",
    "reach_command",
    "reach_position_tracking_exp",
    "reach_success_bonus",
    "reach_base_standoff_penalty",
    "reach_wrong_hand_penalty",
    "reach_pace_exp",
    "reach_stop_at_target",
    "reach_base_overspeed_penalty",
    "reach_command_levels",
]


class ReachTargetCommand(CommandTerm):
    """Single 3D point goal to be touched with a commanded (left/right) hand."""

    cfg: ReachTargetCommandCfg

    def __init__(self, cfg: ReachTargetCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        self._left_body_idx = self._resolve_body_index(cfg.left_wrist_body_name)
        self._right_body_idx = self._resolve_body_index(cfg.right_wrist_body_name)
        self._palm_offset = torch.tensor(
            cfg.palm_offset_b, device=self.device, dtype=torch.float32
        )

        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.hand_sel = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.left_palm_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_palm_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.sel_palm_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.distance = torch.zeros(self.num_envs, device=self.device)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.t_star = torch.zeros(self.num_envs, device=self.device)
        self.elapsed = torch.zeros(self.num_envs, device=self.device)
        self.remaining_time = torch.zeros(self.num_envs, device=self.device)
        self.v_target = torch.zeros(self.num_envs, device=self.device)

        self.arrived = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self._command = torch.zeros(self.num_envs, 7, device=self.device)

        self.metrics["distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["success_rate"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resolve_body_index(self, name: str) -> int:
        body_names = list(self.robot.body_names)
        if name in body_names:
            return body_names.index(name)
        for idx, candidate in enumerate(body_names):
            if candidate.endswith(name) or candidate.endswith("/" + name):
                return idx
        raise ValueError(
            f"Body '{name}' not found in entity '{self.cfg.entity_name}'. "
            f"Available: {body_names}"
        )

    def _palm_positions_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos = self.robot.data.body_link_pos_w
        body_quat = self.robot.data.body_link_quat_w
        offset = self._palm_offset.expand(self.num_envs, 3)
        left = body_pos[:, self._left_body_idx] + quat_apply(
            body_quat[:, self._left_body_idx], offset
        )
        right = body_pos[:, self._right_body_idx] + quat_apply(
            body_quat[:, self._right_body_idx], offset
        )
        return left, right

    def _target_in_heading_frame(self) -> torch.Tensor:
        """Target position relative to the pelvis, rotated into the heading frame."""
        root_pos = self.robot.data.root_link_pos_w
        yaw = self.robot.data.heading_w
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        dx = self.target_pos_w - root_pos
        x_b = cos_y * dx[:, 0] + sin_y * dx[:, 1]
        y_b = -sin_y * dx[:, 0] + cos_y * dx[:, 1]
        z_b = dx[:, 2]
        return torch.stack((x_b, y_b, z_b), dim=-1)

    def _refresh_command(self) -> None:
        self._command[:, :3] = self._target_in_heading_frame()
        self._command[:, 3] = (self.hand_sel == 0).float()
        self._command[:, 4] = (self.hand_sel == 1).float()
        self._command[:, 5] = torch.clamp(
            self.remaining_time / self.cfg.time_clip, 0.0, 1.0
        )
        self._command[:, 6] = self.arrived.float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        self.hand_sel[env_ids] = (r.uniform_(0.0, 1.0) < self.cfg.p_right).long()

        dist = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
        azim = torch.empty(n, device=self.device).uniform_(*self.cfg.angle_range)
        height = torch.empty(n, device=self.device).uniform_(*self.cfg.height_range)

        root_pos = self.robot.data.root_link_pos_w[env_ids]
        yaw = self.robot.data.heading_w[env_ids]
        direction = yaw + azim
        self.target_pos_w[env_ids, 0] = root_pos[:, 0] + dist * torch.cos(direction)
        self.target_pos_w[env_ids, 1] = root_pos[:, 1] + dist * torch.sin(direction)
        self.target_pos_w[env_ids, 2] = height

        v_target = torch.empty(n, device=self.device).uniform_(*self.cfg.v_target_range)
        self.v_target[env_ids] = v_target

        d_xy = torch.norm(root_pos[:, :2] - self.target_pos_w[env_ids, :2], dim=-1)
        self.t_star[env_ids] = (
            torch.clamp(d_xy - self.cfg.standoff_radius, min=0.0) / v_target
        )
        self.elapsed[env_ids] = 0.0
        self.remaining_time[env_ids] = self.t_star[env_ids]
        self.arrived[env_ids] = False

        self._refresh_command()

    def _update_command(self) -> None:
        self.elapsed += self._env.step_dt

        self.left_palm_w, self.right_palm_w = self._palm_positions_w()
        is_right = (self.hand_sel == 1).unsqueeze(-1)
        self.sel_palm_w = torch.where(is_right, self.right_palm_w, self.left_palm_w)
        self.distance = torch.norm(self.sel_palm_w - self.target_pos_w, dim=-1)
        self.reached = self.distance < self.cfg.reach_tol

        root_xy = self.robot.data.root_link_pos_w[:, :2]
        d_nb = torch.clamp(
            torch.norm(root_xy - self.target_pos_w[:, :2], dim=-1)
            - self.cfg.standoff_radius,
            min=0.0,
        )
        base_speed = torch.norm(self.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
        self.arrived |= (d_nb < self.cfg.arrive_radius) & (
            base_speed < self.cfg.arrive_speed
        )

        self.remaining_time = torch.clamp(self.t_star - self.elapsed, min=0.0)
        self._refresh_command()

    def _update_metrics(self) -> None:
        self.metrics["distance"] = self.distance
        self.metrics["success_rate"] = self.reached.float()

    # -- Debug visualization -------------------------------------------------

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        target = self.target_pos_w.cpu().numpy()
        palm = self.sel_palm_w.cpu().numpy()
        reached = self.reached.cpu().numpy()
        for batch in env_indices:
            color = (0.2, 0.9, 0.2, 0.9) if reached[batch] else (0.9, 0.5, 0.1, 0.9)
            visualizer.add_sphere(
                center=target[batch],
                radius=float(self.cfg.reach_tol),
                color=color,
                label=f"reach_target_{batch}",
            )
            visualizer.add_arrow(
                palm[batch],
                target[batch],
                color=(0.1, 0.4, 0.9, 0.6),
                width=0.01,
                label=f"reach_vec_{batch}",
            )


@dataclass(kw_only=True)
class ReachTargetCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"
    palm_offset_b: tuple[float, float, float] = (0.08, 0.0, 0.0)

    p_right: float = 0.5
    """Probability the right hand is the commanded hand."""

    distance_range: tuple[float, float] = (0.0, 2.0)
    """Horizontal pelvis->target spawn distance (m). Widened by curriculum."""

    angle_range: tuple[float, float] = (-3.14159, 3.14159)
    """Azimuth of the target relative to the current heading (rad)."""

    height_range: tuple[float, float] = (1.05, 1.30)
    """Target height band (m, world z). Straddles shoulder height (~1.085 m) and
  stays comfortably inside the FK full-extension palm reach (<=1.56 m) so a
  slightly-bent arm reaches it. Anti-body-bump is handled by the standoff."""

    reach_tol: float = 0.12
    """Palm-to-target distance counted as a touch (m)."""

    standoff_radius: float = 0.30
    """Pelvis-to-target horizontal distance below which a soft anti-body-bump
  penalty applies (m)."""

    v_target_range: tuple[float, float] = (0.6, 2.0)

    time_clip: float = 3.0

    arrive_radius: float = 0.20
    """Pelvis->neighbourhood distance (m) below which, combined with low speed,
  the phase latches to REACH. Small so the body is genuinely at the standoff
  shell before the hand reward unlocks."""

    arrive_speed: float = 0.30
    """Base speed (m/s) below which the robot counts as 'arrived' (slow). Forces
  a decelerate-before-arriving: arriving fast does not latch the reach phase."""

    def build(self, env: ManagerBasedRlEnv) -> ReachTargetCommand:
        return ReachTargetCommand(self, env)


def reach_command(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return the heading-frame target + hand one-hot ``(x, y, z, left, right)``."""
    command = env.command_manager.get_command(command_name)
    assert command is not None, f"Command '{command_name}' not found."
    return command


def _reach_command(env: ManagerBasedRlEnv, command_name: str) -> ReachTargetCommand:
    command = env.command_manager.get_term(command_name)
    if isinstance(command, ReachTargetCommand):
        return command
    raise TypeError(
        f"Command '{command_name}' is not a ReachTargetCommand. Got: {type(command)}"
    )


def reach_position_tracking_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
    phase_gated: bool = False,
) -> torch.Tensor:
    command = _reach_command(env, command_name)
    reward = torch.exp(-torch.square(command.distance) / std**2)
    if phase_gated:
        reward = reward * command.arrived.float()
    return reward


def reach_success_bonus(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Sparse +1 when the commanded hand is on target and the base is not
    bumping it (pelvis outside the standoff radius)."""
    command = _reach_command(env, command_name)
    root_xy = command.robot.data.root_link_pos_w[:, :2]
    base_dist = torch.norm(root_xy - command.target_pos_w[:, :2], dim=-1)
    outside_standoff = base_dist > command.cfg.standoff_radius
    return (command.reached & outside_standoff & command.arrived).float()


def reach_base_standoff_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = _reach_command(env, command_name)
    root_xy = command.robot.data.root_link_pos_w[:, :2]
    base_dist = torch.norm(root_xy - command.target_pos_w[:, :2], dim=-1)
    return torch.clamp(command.cfg.standoff_radius - base_dist, min=0.0)


def reach_wrong_hand_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = _reach_command(env, command_name)
    is_right = (command.hand_sel == 1).unsqueeze(-1)
    wrong_palm = torch.where(is_right, command.left_palm_w, command.right_palm_w)
    wrong_dist = torch.norm(wrong_palm - command.target_pos_w, dim=-1)
    return (wrong_dist < command.cfg.reach_tol).float()


def reach_pace_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Reward the pelvis for tracking the time schedule to the neighbourhood.

    On schedule, the remaining horizontal pelvis->neighbourhood distance should
    equal ``v_target * remaining_time`` (``v_target`` is the per-target sampled
    speed). We reward ``exp(-(d_nb - v_target * remaining)^2 / std^2)``, which:

    * gives the *body* its own dense objective (so it walks/runs to the target
      instead of being dragged there by the reaching hand), and
    * pins the travel speed to the *budgeted* ``v_target`` -- going faster
      drives ``d_nb`` below the schedule and loses reward (so it slows down /
      decelerates before arriving), going slower falls behind. The gait the
      policy picks therefore follows the time budget: tight -> run, loose ->
      walk. This also counters the discount-factor incentive to always rush.

    Once arrived on time (``d_nb = 0`` and ``remaining = 0``) the term
    saturates at 1 and stops interfering with the hand touch.
    """
    command = _reach_command(env, command_name)
    root_xy = command.robot.data.root_link_pos_w[:, :2]
    d_nb = torch.clamp(
        torch.norm(root_xy - command.target_pos_w[:, :2], dim=-1)
        - command.cfg.standoff_radius,
        min=0.0,
    )
    scheduled = command.v_target * command.remaining_time
    err = d_nb - scheduled
    return torch.exp(-torch.square(err) / std**2)


def reach_stop_at_target(
    env: ManagerBasedRlEnv,
    command_name: str,
    speed_std: float,
) -> torch.Tensor:
    command = _reach_command(env, command_name)
    speed = torch.norm(command.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
    return command.arrived.float() * torch.exp(-torch.square(speed) / speed_std**2)


def reach_base_overspeed_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
    max_speed: float,
) -> torch.Tensor:
    command = _reach_command(env, command_name)
    speed = torch.norm(command.robot.data.root_link_lin_vel_b[:, :2], dim=-1)
    return torch.clamp(speed - max_speed, min=0.0)


class ReachLevelStage(TypedDict, total=False):
    step: int
    distance_range: tuple[float, float]
    angle_range: tuple[float, float]
    v_target_range: tuple[float, float]


def reach_command_levels(
    env: ManagerBasedRlEnv,
    env_ids: torch.Tensor | slice | None,
    command_name: str,
    stages: list[ReachLevelStage],
) -> dict[str, float]:
    del env_ids
    command = cast(ReachTargetCommand, env.command_manager.get_term(command_name))
    distance_range = command.cfg.distance_range
    angle_range = command.cfg.angle_range
    v_target_range = command.cfg.v_target_range
    for stage in stages:
        if env.common_step_counter > stage["step"]:
            distance_range = stage["distance_range"]
            angle_range = stage["angle_range"]
            if "v_target_range" in stage:
                v_target_range = stage["v_target_range"]
    command.cfg.distance_range = distance_range
    command.cfg.angle_range = angle_range
    command.cfg.v_target_range = v_target_range
    return {
        "distance_max": float(distance_range[1]),
        "angle_max": float(angle_range[1]),
        "v_target_max": float(v_target_range[1]),
    }
