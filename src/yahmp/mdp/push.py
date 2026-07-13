from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, TypedDict, cast

import numpy as np
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
    "push_speed_limit",
    "push_position_tracking_exp",
    "push_hands_on_crate",
    "push_hands_contact",
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
        self.metrics["hands_gate"] = torch.zeros(self.num_envs, device=self.device)
        # Goals completed this episode (multi-goal). Kept directly in ``metrics``:
        # the base ``reset()`` logs its per-episode mean and zeroes it for us.
        self.metrics["goals_reached"] = torch.zeros(self.num_envs, device=self.device)
        self.just_reached = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )

        # Hand bodies (both wrists) whose distance to the crate SURFACE drives the
        # gate: crate-outcome rewards (progress/position/success) only pay when a
        # hand is on the crate, so pushing with the body earns ~nothing.
        hand_ids, _ = self.robot.find_bodies([r".*_wrist_yaw_link"])
        self.hand_body_ids = hand_ids
        self._half_extents = torch.tensor(cfg.crate_half_extents, device=self.device)
        self.hands_surface_dist = torch.zeros(
            self.num_envs, len(hand_ids), device=self.device
        )
        self.hands_gate = torch.zeros(self.num_envs, device=self.device)

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

    def _sample_goal(
        self, env_ids: torch.Tensor, crate_xy: torch.Tensor, base_dir: torch.Tensor
    ) -> None:
        """Place a new goal at (crate_xy + dist along base_dir+azim) and re-arm the
        progress bookkeeping. ``crate_xy``/``base_dir`` are supplied by the caller
        because they differ between the reset path (deterministic pose, nominal +x)
        and the mid-episode reach path (fresh crate pose, robot->crate direction)."""
        n = len(env_ids)
        dist = torch.empty(n, device=self.device).uniform_(*self.cfg.distance_range)
        azim = torch.empty(n, device=self.device).uniform_(*self.cfg.angle_range)

        direction = base_dir + azim
        self.target_pos_w[env_ids, 0] = crate_xy[:, 0] + dist * torch.cos(direction)
        self.target_pos_w[env_ids, 1] = crate_xy[:, 1] + dist * torch.sin(direction)

        new_dist = torch.linalg.norm(crate_xy - self.target_pos_w[env_ids], dim=-1)
        self.distance[env_ids] = new_dist
        self.prev_distance[env_ids] = new_dist
        self.progress[env_ids] = 0.0

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        """RESET path: sample the first goal of the episode.

        Crate reset position (world), DETERMINISTIC: default_root_state (=
        init_state, env-local) + env_origins. We must NOT read
        ``crate.data.root_com_pos_w`` here: this runs inside the reset event,
        where derived quantities are still STALE (the drifted end-of-episode
        crate pose), which would anchor the goal to the wrong spot. The crate
        is re-placed to exactly this pose by the ``reset_crate`` event.
        Likewise the robot's fresh yaw is unavailable, so the base direction is
        the nominal forward +x (the constrained base reset faces ~+x).
        """
        origins = self._env.scene.env_origins[env_ids, :2]
        crate_xy = self.crate.data.default_root_state[env_ids, :2] + origins
        base_dir = torch.zeros(len(env_ids), device=self.device)
        self._sample_goal(env_ids, crate_xy, base_dir)

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

        ## MULTI-GOAL: on reach, immediately sample the next goal (episode keeps
        ## going). Structural anti-dawdling: completing a goal re-opens the
        ## progress/success income, so parking the crate near the goal is never
        ## the return-maximising strategy. ``just_reached`` stays True for this
        ## step so ``push_success_bonus`` still pays once. Unlike the reset path,
        ## HERE the derived quantities are fresh (we run after the physics step),
        ## so we anchor on the crate's ACTUAL pose and continue pushing away from
        ## the robot (robot->crate direction + curriculum azimuth).
        if bool(self.just_reached.any()):
            ids = self.just_reached.nonzero(as_tuple=False).squeeze(-1)
            crate_xy = pos_box_w[ids, :2]
            base_dir = torch.atan2(dy[ids], dx[ids])  # robot -> crate (world)
            self._sample_goal(ids, crate_xy, base_dir)
            self.metrics["goals_reached"][ids] += 1.0
            # Refresh the goal-dependent command entries for the resampled envs so
            # the observation reflects the NEW goal already this step.
            gx = self.target_pos_w[ids, 0] - pos_box_w[ids, 0]
            gy = self.target_pos_w[ids, 1] - pos_box_w[ids, 1]
            self._command[ids, 2] = cos_y[ids] * gx + sin_y[ids] * gy
            self._command[ids, 3] = -sin_y[ids] * gx + cos_y[ids] * gy
            self._command[ids, 4] = self.distance[ids]

        ## Hand -> crate SURFACE distance, and the soft gate derived from it.
        # Bring each hand into the crate's local frame (root_link_quat_w is
        # body->world, so quat_apply_inverse gives world->box), clamp to the box
        # half-extents to get the closest point on the box, then measure the
        # residual = true distance to the surface (0 if the hand is inside).
        hands_w = self.robot.data.body_link_pos_w[:, self.hand_body_ids]  # (N,H,3)
        box_pos = self.crate.data.root_link_pos_w  # (N,3)
        box_quat = self.crate.data.root_link_quat_w  # (N,4)
        n_hands = hands_w.shape[1]
        rel = hands_w - box_pos.unsqueeze(1)  # (N,H,3)
        q = box_quat.unsqueeze(1).expand(-1, n_hands, -1).reshape(-1, 4)
        local = quat_apply_inverse(q, rel.reshape(-1, 3)).reshape(hands_w.shape)
        clamped = torch.clamp(local, -self._half_extents, self._half_extents)
        self.hands_surface_dist = torch.linalg.norm(local - clamped, dim=-1)  # (N,H)
        d_min = self.hands_surface_dist.min(dim=-1).values  # (N,)
        self.hands_gate = torch.exp(-torch.square(d_min) / (self.cfg.gate_std**2))

        ##Oppure con i quaternioni:
        # pos_robot_w = self.robot.data.root_com_pos_w  # (N, 3)
        # yaw_q = yaw_quat(self.robot.data.root_com_quat_w)  # (N, 4)

        # delta_crate = pos_box_w - pos_robot_w  # (N, 3)
        # delta_target = torch.zeros_like(delta_crate)
        # delta_target[:, :2] = self.target_pos_w - pos_box_w[:, :2]
        # delta = pos_box_w[:, :2] - self.target_pos_w

    def _update_metrics(self) -> None:
        self.metrics["distance"] = self.distance
        self.metrics["hands_gate"] = self.hands_gate

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        """Draw the goal (where the crate must end up) as a sphere, plus a segment
        from the crate to the goal (the remaining push vector). Green once reached,
        red while still to push."""
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        z = float(self._half_extents[2])  # crate-centre height
        for batch in env_indices:
            goal = self.target_pos_w[batch].detach().cpu().numpy()  # (2,)
            goal_xyz = np.array([goal[0], goal[1], z], dtype=np.float32)
            reached = bool(self.just_reached[batch].item())
            color = (0.1, 0.9, 0.1, 0.6) if reached else (0.9, 0.2, 0.2, 0.6)
            visualizer.add_sphere(
                center=goal_xyz,
                radius=0.12,
                color=color,
                label=f"push_goal_{batch}",
            )
            # Remaining push vector (skip the degenerate zero-length case).
            if float(self.distance[batch].item()) > 0.05:
                crate_xyz = (
                    self.crate.data.root_link_pos_w[batch].detach().cpu().numpy()
                )
                visualizer.add_cylinder(
                    start=crate_xyz,
                    end=goal_xyz,
                    radius=0.02,
                    color=(0.95, 0.85, 0.1, 0.6),
                    label=f"push_vec_{batch}",
                )


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
    # Crate box half-extents (MUST match the crate geom ``size`` in
    # ``_crate_spec``); used to measure hand->surface distance for the gate.
    crate_half_extents: tuple[float, float, float] = (0.3, 0.3, 0.4)
    # Width of the soft gate on hand->crate-surface distance. A hand this far
    # from the surface earns ~37% of the gated (crate-outcome) rewards.
    gate_std: float = 0.35
    # Draw the goal marker + crate->goal segment in the viewer by default.
    debug_vis: bool = True

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
    """Crate-progress towards the goal, GATED by hand contact: moving the crate
    only pays while a hand is on it (pushing with the body earns ~nothing)."""
    command = _push_command(env, command_name)
    raw = torch.clamp(command.progress / max(env.step_dt, 1e-6), -cap, cap)
    return raw * command.hands_gate


def push_success_bonus(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Bonus for reaching the goal, GATED by hand contact (a hand must be on the
    crate at the reach moment)."""
    command = _push_command(env, command_name)
    return command.just_reached.float() * command.hands_gate


def push_velocity_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float, cruise_speed: float = 1.1
) -> torch.Tensor:
    ## Push the robot to walk rather than trying to maximize the return always running
    command = _push_command(env, command_name)
    speed = torch.linalg.norm(
        command.robot.data.root_com_lin_vel_w[..., :2], dim=-1
    )  # (N,)
    return torch.exp(-torch.square(speed - cruise_speed) / (std**2))


def push_speed_limit(
    env: ManagerBasedRlEnv, command_name: str, v_max: float = 1.2
) -> torch.Tensor:
    """Walk-not-run as a PENALTY (not a reward): 0 while walking/standing, negative
    only above ``v_max`` (discourages the running primitives). Unlike a target-speed
    bowl this pays nothing for moving, so it can't be farmed by circling. Use with a
    positive weight (the value is <= 0)."""
    command = _push_command(env, command_name)
    speed = torch.linalg.norm(command.robot.data.root_com_lin_vel_w[..., :2], dim=-1)
    excess = torch.clamp(speed - v_max, min=0.0)
    return -torch.square(excess)


def push_position_tracking_exp(
    env: ManagerBasedRlEnv, command_name: str, std: float
) -> torch.Tensor:
    """Crate-near-goal bowl, GATED by hand contact (same anti-bulldoze logic as
    ``push_progress``)."""
    command = _push_command(env, command_name)
    return torch.exp(-torch.square(command.distance) / (std**2)) * command.hands_gate


def reached_termination(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
    """Return a boolean tensor indicating whether the goal has been reached."""
    command = _push_command(env, command_name)
    return command.distance < command.cfg.reach_tol


def push_hands_on_crate(
    env: ManagerBasedRlEnv, command_name: str, std: float = 0.35
) -> torch.Tensor:
    """Dense shaping: pull the hands onto the crate SURFACE. Uses the hand->surface
    distance computed once per step in the command (single source of truth, shared
    with the gate). Ungated so it can guide the hands in before the gate opens."""
    command = _push_command(env, command_name)
    return torch.exp(-torch.square(command.hands_surface_dist) / (std**2)).mean(dim=-1)


def push_hands_contact(
    env: ManagerBasedRlEnv, sensor_name: str = "hands_crate_contact"
) -> torch.Tensor:
    """Fraction of hands (0, 0.5, 1) in actual physical contact with the crate,
    read from the ``hands_crate_contact`` contact sensor. The true physical signal
    (can't be gamed by hovering) and a clean wandb metric."""
    sensor = env.scene[sensor_name]
    found = sensor.data.found  # (N, H)
    assert found is not None
    return (found > 0).float().mean(dim=-1)


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
        # ``>=`` so the step-0 stage applies from the very first curriculum call
        # (``common_step_counter == 0``); with ``>`` the first resets would leak
        # the cfg-default (far) distance_range before the near-first stage kicks in.
        if env.common_step_counter >= stage["step"]:
            distance_range = stage["distance_range"]
            angle_range = stage["angle_range"]
    command.cfg.distance_range = distance_range
    command.cfg.angle_range = angle_range
    return {
        "distance_max": float(distance_range[1]),
        "angle_max": float(angle_range[1]),
    }
