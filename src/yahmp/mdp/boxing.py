from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

import torch
from mjlab.entity import Entity
from mjlab.managers.command_manager import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv
    from mjlab.viewer.debug_visualizer import DebugVisualizer

__all__ = [
    "BoxingStrikeCommand",
    "BoxingStrikeCommandCfg",
    "boxing_command",
    "strike_homing_exp",
    "strike_success_bonus",
    "strike_impact_bonus",
    "strike_idle_hand_home_penalty",
    "strike_wrong_hand_penalty",
]


class BoxingStrikeCommand(CommandTerm):
    """Floating in-front strike target to be hit with a commanded hand."""

    cfg: BoxingStrikeCommandCfg

    def __init__(self, cfg: BoxingStrikeCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        self.robot: Entity = env.scene[cfg.entity_name]

        self._left_body_idx = self._resolve_body_index(cfg.left_wrist_body_name)
        self._right_body_idx = self._resolve_body_index(cfg.right_wrist_body_name)
        self._palm_offset = torch.tensor(
            cfg.palm_offset_b, device=self.device, dtype=torch.float32
        )

        self._left_arm_joint_idx = self._resolve_arm_joints("left")
        self._right_arm_joint_idx = self._resolve_arm_joints("right")

        self.offset_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self.height = torch.zeros(self.num_envs, device=self.device)
        self.hand_sel = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)

        self.target_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.left_palm_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.right_palm_w = torch.zeros(self.num_envs, 3, device=self.device)
        self.sel_palm_w = torch.zeros(self.num_envs, 3, device=self.device)

        self.sel_palm_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_sel_palm_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.sel_palm_speed = torch.zeros(self.num_envs, device=self.device)
        self.distance = torch.zeros(self.num_envs, device=self.device)
        self.reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

        self.just_struck = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        # Cumulative strikes per episode (for the metric).
        self.strike_count = torch.zeros(self.num_envs, device=self.device)

        # Observed command: [target_x_b, target_y_b, target_z_b, is_left, is_right].
        self._command = torch.zeros(self.num_envs, 5, device=self.device)

        self.metrics["distance"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["strike_count"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _resolve_arm_joints(self, side: str) -> torch.Tensor:
        idx = [
            i
            for i, n in enumerate(self.robot.joint_names)
            if (f"{side}_shoulder" in n or f"{side}_elbow" in n or f"{side}_wrist" in n)
        ]
        if not idx:
            raise ValueError(
                f"No '{side}' arm joints found in {list(self.robot.joint_names)}"
            )
        return torch.tensor(idx, device=self.device, dtype=torch.long)

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

    def _recompute_target_w(self) -> None:
        """World target = base xy + R(yaw) @ heading-frame offset, fixed height."""
        root_pos = self.robot.data.root_link_pos_w
        yaw = self.robot.data.heading_w
        cos_y = torch.cos(yaw)
        sin_y = torch.sin(yaw)
        off_x = self.offset_xy[:, 0]
        off_y = self.offset_xy[:, 1]
        self.target_pos_w[:, 0] = root_pos[:, 0] + cos_y * off_x - sin_y * off_y
        self.target_pos_w[:, 1] = root_pos[:, 1] + sin_y * off_x + cos_y * off_y
        self.target_pos_w[:, 2] = self.height

    def _refresh_command(self) -> None:
        root_z = self.robot.data.root_link_pos_w[:, 2]
        self._command[:, 0] = self.offset_xy[:, 0]
        self._command[:, 1] = self.offset_xy[:, 1]
        self._command[:, 2] = self.height - root_z
        self._command[:, 3] = (self.hand_sel == 0).float()
        self._command[:, 4] = (self.hand_sel == 1).float()

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        n = len(env_ids)
        if n == 0:
            return
        r = torch.empty(n, device=self.device)
        hand = (r.uniform_(0.0, 1.0) < self.cfg.p_right).long()  # 1 = right
        self.hand_sel[env_ids] = hand

        # Forward offset (always in front).
        off_x = torch.empty(n, device=self.device).uniform_(*self.cfg.forward_range)
        # Lateral offset biased to the commanded hand's side: the outboard reach
        # is on that side, with a small inboard (cross-body) allowance.
        out = self.cfg.lateral_outboard
        inb = self.cfg.lateral_inboard
        # right hand -> y in [-out, +inb]; left hand -> y in [-inb, +out].
        lo = torch.where(
            hand == 1, torch.full_like(off_x, -out), torch.full_like(off_x, -inb)
        )
        hi = torch.where(
            hand == 1, torch.full_like(off_x, inb), torch.full_like(off_x, out)
        )
        off_y = lo + (hi - lo) * torch.rand(n, device=self.device)

        self.offset_xy[env_ids, 0] = off_x
        self.offset_xy[env_ids, 1] = off_y
        self.height[env_ids] = torch.empty(n, device=self.device).uniform_(
            *self.cfg.height_range
        )
        self._recompute_target_w()
        self._refresh_command()

    def _update_command(self) -> None:
        self._recompute_target_w()
        self.left_palm_w, self.right_palm_w = self._palm_positions_w()
        is_right = (self.hand_sel == 1).unsqueeze(-1)
        self.sel_palm_w = torch.where(is_right, self.right_palm_w, self.left_palm_w)

        root_pos = self.robot.data.root_link_pos_w
        yaw = self.robot.data.heading_w
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)
        d = self.sel_palm_w - root_pos
        palm_b = torch.stack(
            (
                cos_y * d[:, 0] + sin_y * d[:, 1],
                -sin_y * d[:, 0] + cos_y * d[:, 1],
                d[:, 2],
            ),
            dim=-1,
        )
        self.sel_palm_speed = torch.norm(palm_b - self.prev_sel_palm_b, dim=-1) / max(
            self._env.step_dt, 1e-6
        )
        self.prev_sel_palm_b = palm_b
        self.sel_palm_b = palm_b

        self.distance = torch.norm(self.sel_palm_w - self.target_pos_w, dim=-1)
        self.reached = self.distance < self.cfg.reach_tol

        self.just_struck = self.reached.clone()
        struck_ids = self.reached.nonzero().flatten()
        if len(struck_ids) > 0:
            self.strike_count[struck_ids] += 1.0
            self._resample(struck_ids)  # new offset+hand AND reset timer
            self._recompute_target_w()
            self.distance = torch.norm(self.sel_palm_w - self.target_pos_w, dim=-1)
        self._refresh_command()

    def _update_metrics(self) -> None:
        self.metrics["distance"] = self.distance
        self.metrics["strike_count"] = self.strike_count

    def _debug_vis_impl(self, visualizer: "DebugVisualizer") -> None:
        env_indices = visualizer.get_env_indices(self.num_envs)
        if not env_indices:
            return
        target = self.target_pos_w.cpu().numpy()
        palm = self.sel_palm_w.cpu().numpy()
        reached = self.reached.cpu().numpy()
        for batch in env_indices:
            color = (0.2, 0.9, 0.2, 0.9) if reached[batch] else (0.9, 0.2, 0.2, 0.9)
            visualizer.add_sphere(
                center=target[batch],
                radius=float(self.cfg.reach_tol),
                color=color,
                label=f"strike_target_{batch}",
            )
            visualizer.add_arrow(
                palm[batch],
                target[batch],
                color=(0.1, 0.4, 0.9, 0.6),
                width=0.01,
                label=f"strike_vec_{batch}",
            )


@dataclass(kw_only=True)
class BoxingStrikeCommandCfg(CommandTermCfg):
    entity_name: str = "robot"
    left_wrist_body_name: str = "left_wrist_yaw_link"
    right_wrist_body_name: str = "right_wrist_yaw_link"
    palm_offset_b: tuple[float, float, float] = (0.08, 0.0, 0.0)

    p_right: float = 0.5
    """Probability the right hand is the commanded hand for a target."""

    forward_range: tuple[float, float] = (0.30, 0.55)
    """In-front (heading +x) offset of the bag (m). Arm-reach punching range."""

    lateral_outboard: float = 0.30
    """Max lateral offset on the commanded hand's side (m, heading +/-y)."""

    lateral_inboard: float = 0.10
    """Max cross-body (inboard) lateral offset for the commanded hand (m)."""

    height_range: tuple[float, float] = (1.05, 1.45)
    """World-z height band of the bag (m): chest-to-head, inside FK reach."""

    reach_tol: float = 0.15
    """Palm-to-target distance counted as a strike (m). A touch resamples."""

    def build(self, env: ManagerBasedRlEnv) -> BoxingStrikeCommand:
        return BoxingStrikeCommand(self, env)


def boxing_command(
    env: ManagerBasedRlEnv,
    velocity_command_name: str,
    strike_command_name: str,
) -> torch.Tensor:
    """Combined goal term: ``[vx, vy, wz | tx_b, ty_b, tz_b, is_left, is_right]``.

    Concatenated into a SINGLE obs term named ``command`` so the hierarchical
    runner derives the actor goal slice from it (the runner reads exactly one
    obs term called ``command``)."""
    vel = env.command_manager.get_command(velocity_command_name)
    strike = env.command_manager.get_command(strike_command_name)
    return torch.cat((vel, strike), dim=-1)


def _strike_command(env: ManagerBasedRlEnv, command_name: str) -> BoxingStrikeCommand:
    command = env.command_manager.get_term(command_name)
    if isinstance(command, BoxingStrikeCommand):
        return command
    raise TypeError(
        f"Command '{command_name}' is not a BoxingStrikeCommand. Got: {type(command)}"
    )


def strike_homing_exp(
    env: ManagerBasedRlEnv,
    command_name: str,
    std: float,
) -> torch.Tensor:
    """Dense reward pulling the *commanded* hand to the bag: exp(-d^2/std^2)."""
    command = _strike_command(env, command_name)
    return torch.exp(-torch.square(command.distance) / std**2)


def strike_success_bonus(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Sparse +1 on the step the commanded hand strikes the bag (pre-resample)."""
    command = _strike_command(env, command_name)
    return command.just_struck.float()


def strike_impact_bonus(
    env: ManagerBasedRlEnv,
    command_name: str,
    speed_cap: float = 3.0,
) -> torch.Tensor:
    """Reward a FAST strike: on the landing step, the arm-relative palm speed
    (clamped to ``speed_cap``). Rewards a "snap" (rest -> explosive jab) rather
    than a slow drift into the bag; uses base-frame speed so walking faster
    cannot farm it."""
    command = _strike_command(env, command_name)
    return command.just_struck.float() * torch.clamp(
        command.sel_palm_speed, 0.0, speed_cap
    )


def strike_idle_hand_home_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    command = _strike_command(env, command_name)
    qpos = command.robot.data.joint_pos
    qdef = command.robot.data.default_joint_pos
    err2 = torch.square(qpos - qdef)
    left_err = err2[:, command._left_arm_joint_idx].mean(dim=-1)
    right_err = err2[:, command._right_arm_joint_idx].mean(dim=-1)
    is_right_cmd = command.hand_sel == 1
    # Idle arm = the one NOT commanded.
    return torch.where(is_right_cmd, left_err, right_err)


def strike_wrong_hand_penalty(
    env: ManagerBasedRlEnv,
    command_name: str,
) -> torch.Tensor:
    """Narrow backstop: +1 when the *non*-commanded hand is on the bag. Use with
    a negative weight so the policy strikes with the commanded hand."""
    command = _strike_command(env, command_name)
    is_right = (command.hand_sel == 1).unsqueeze(-1)
    wrong_palm = torch.where(is_right, command.left_palm_w, command.right_palm_w)
    wrong_dist = torch.norm(wrong_palm - command.target_pos_w, dim=-1)
    return (wrong_dist < command.cfg.reach_tol).float()
