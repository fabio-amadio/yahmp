"""Play a locomotion checkpoint with a scripted velocity-command schedule.

At every env.step() the "twist" command term is overwritten with the value
from the SCHEDULE table defined below — bypassing UniformVelocityCommand's
random sampling and the joystick GUI. Edit SCHEDULE to drive the robot
through specific maneuvers (walk -> run -> turn in run -> ...). Phase
transitions are printed to stdout so it's clear which command is active.

The schedule loops once its total duration elapses.

Usage:
  uv run python scripts/play_scripted.py Mjlab-YAHMP-Locomotion-Unitree-G1 \\
      --checkpoint-file logs/rsl_rl/g1_yahmp_locomotion_lafan/<run>/model_9000.pt \\
      --imitation-checkpoint-file /path/to/imitation.pt \\
      --num-envs 1
"""

from __future__ import annotations

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mjlab
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    _REGISTRY,
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

SCHEDULE: list[tuple[float, float, float, float, str]] = [
    (5.0, 0.0, 0.0, 0.0, "standing"),
    #
    # (6.0, 0.4, 0.0, 0.0, "walk medio"),
    (6.0, 1.0, 0.0, 0.0, "walk (vx=1.0)"),
    (6.0, 1.5, 0.0, 0.0, "walk (vx=1.2)"),
    (6.0, 2.0, 0.0, 0.0, "run (vx=2.0)"),
    (6.0, 2.5, 0.0, 0.5, "run turning (vx=2.5)"),
    (5.0, 2.5, 0.0, -1.2, "decel (vx=1.3)"),
    (6.0, 3.0, 0.0, 0.0, "decel (vx=1.3)"),
    (8.0, -1.0, 0.0, 0.0, "walk backward (vx=-1.5)"),
    (1.5, 0.0, 0.0, 0.0, "walk backward (vx=-1.5)"),
    (6.0, 0.0, 0.0, 1.0, "decel (vx=0.0)"),
    (6.0, 0.0, 0.0, -1.2, "decel (vx=-1.0)"),
    (6.0, 0.0, 1.0, 0.0, "lateral (v7=-0.7)"),
    # (6.0, 0.0, -1.0, 0.0, "lateral (v7=0.7)"),
    (10.0, 2.5, 0.0, 0.0, "running"),
    (10.0, 3.0, 0.0, 0.0, "running"),
    (6.0, 3.5, 0.0, 0.0, "running"),
    # (6.0, 2.0, 0.0, 0.0, "running"),
]


@dataclass(frozen=True)
class Config:
    checkpoint_file: str
    """Path to the trained locomotion .pt checkpoint."""
    imitation_checkpoint_file: str | None = None
    """Optional imitation .pt (RVQ low-level) checkpoint; required for the YAHMP runner."""
    rvq_num_active_quantizers: int | None = None
    """Active RVQ codebooks; MUST match the trained checkpoint (None = all 8)."""
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = True
    """Disable terminations for a clean demo (the schedule keeps running)."""
    motion_file: str | None = None
    """Unused here, kept for parity with other tasks that need a motion."""

    video: bool = False
    """Record an mp4 alongside the on-screen viewer."""
    video_length: int = 1500
    """Frames (= env steps) to record. ~30 s at 50 Hz control."""
    video_start_step: int = 0
    """Steps to wait before recording begins, to skip the spawn transient."""
    video_height: int | None = 1080
    video_width: int | None = 1920
    # Camera framing for the recorded mp4 (also sets the on-screen start pose).
    cam_elevation: float | None = None
    """Negative = above looking down. Try -25 for a three-quarter gait view."""
    cam_distance: float | None = None
    """Metres from the tracked base. Try 3-3.5 to frame the robot."""
    cam_azimuth: float | None = None
    """Viewing angle (degrees) around the robot."""
    show_twist_arrow: bool = True
    """Draw the twist command's velocity arrow at the base. Off for clean demos."""


def _phase_at(t: float, total: float) -> tuple[float, float, float, str]:
    """Return (vx, vy, wz, label) at simulated time t (looping)."""
    if total <= 0.0:
        dur, vx, vy, wz, lbl = SCHEDULE[-1]
        return vx, vy, wz, lbl
    t_mod = t % total
    acc = 0.0
    for dur, vx, vy, wz, lbl in SCHEDULE:
        if t_mod < acc + dur:
            return vx, vy, wz, lbl
        acc += dur
    dur, vx, vy, wz, lbl = SCHEDULE[-1]
    return vx, vy, wz, lbl


def _find_twist_term_name(active_terms: list[str]) -> str:
    for candidate in ("twist", "base_velocity", "velocity"):
        if candidate in active_terms:
            return candidate
    for name in active_terms:
        if "twist" in name.lower() or "velocity" in name.lower():
            return name
    raise RuntimeError(
        f"Could not locate a twist/velocity command term among {active_terms!r}."
    )


def main() -> None:
    configure_torch_backends()

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )

    cfg = tyro.cli(
        Config,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
    print(f"[INFO] Locomotion checkpoint: {checkpoint_path}")

    if cfg.imitation_checkpoint_file is not None:
        registered_cfg = _REGISTRY[chosen_task].rl_cfg
        if not hasattr(registered_cfg, "imitation_checkpoint_file"):
            raise ValueError(
                f"Task {chosen_task} runner cfg does not accept `imitation_checkpoint_file`."
            )
        registered_cfg.imitation_checkpoint_file = cfg.imitation_checkpoint_file
        print(f"[INFO] Imitation checkpoint: {cfg.imitation_checkpoint_file}")

    if cfg.rvq_num_active_quantizers is not None:
        registered_cfg = _REGISTRY[chosen_task].rl_cfg
        if not hasattr(registered_cfg, "rvq_num_active_quantizers"):
            raise ValueError(
                f"Task {chosen_task} runner cfg does not accept "
                "`rvq_num_active_quantizers`."
            )
        registered_cfg.rvq_num_active_quantizers = cfg.rvq_num_active_quantizers
        print(f"[INFO] Active RVQ codebooks: {cfg.rvq_num_active_quantizers}")

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(chosen_task, play=True)
    agent_cfg = load_rl_cfg(chosen_task)

    total_schedule_s = sum(row[0] for row in SCHEDULE)
    # Stretch the episode so a single loop of the schedule plays without auto-reset.
    desired_episode_s = max(total_schedule_s + 2.0, env_cfg.episode_length_s)
    if desired_episode_s != env_cfg.episode_length_s:
        env_cfg.episode_length_s = desired_episode_s
    print(
        f"[INFO] episode_length_s = {desired_episode_s:.1f}s "
        f"(schedule total = {total_schedule_s:.1f}s)"
    )

    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled (no_terminations=True)")

    env_cfg.scene.num_envs = cfg.num_envs

    # Camera framing: applies to both the on-screen viewer and the recorded mp4.
    if cfg.cam_elevation is not None:
        env_cfg.viewer.elevation = cfg.cam_elevation
    if cfg.cam_distance is not None:
        env_cfg.viewer.distance = cfg.cam_distance
    if cfg.cam_azimuth is not None:
        env_cfg.viewer.azimuth = cfg.cam_azimuth
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if cfg.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if cfg.video:
        from mjlab.utils.wrappers import VideoRecorder

        video_folder = checkpoint_path.parent / "videos" / "scripted"
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda s: s == cfg.video_start_step,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print(
            f"[INFO] Recording mp4 -> {video_folder} "
            f"(starts at step {cfg.video_start_step}, {cfg.video_length} frames). "
            "The offscreen camera follows env_cfg.viewer (cam-* flags), not the mouse."
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    cmd_mgr = env.unwrapped.command_manager
    twist_name = _find_twist_term_name(cmd_mgr.active_terms)
    print(f"[INFO] Overriding command term: {twist_name!r}")
    twist_term = cmd_mgr.get_term(twist_name)

    if hasattr(twist_term, "cfg"):
        if hasattr(twist_term.cfg, "rel_standing_envs"):
            twist_term.cfg.rel_standing_envs = 0.0
        if hasattr(twist_term.cfg, "rel_heading_envs"):
            twist_term.cfg.rel_heading_envs = 0.0
        if hasattr(twist_term.cfg, "init_velocity_prob"):
            twist_term.cfg.init_velocity_prob = 0.0
        # Hide the base-frame velocity arrow unless explicitly requested.
        twist_term.cfg.debug_vis = cfg.show_twist_arrow
    if hasattr(twist_term, "is_standing_env"):
        twist_term.is_standing_env[:] = False
    if hasattr(twist_term, "is_heading_env"):
        twist_term.is_heading_env[:] = False
    if hasattr(twist_term, "_joystick_enabled"):
        twist_term._joystick_enabled = None

    step_dt = float(env.unwrapped.step_dt)
    state = {"step_idx": 0, "label": None}
    original_compute = twist_term.compute  # bound method

    def scripted_compute(dt: float) -> None:
        original_compute(dt)
        if hasattr(twist_term, "is_standing_env"):
            twist_term.is_standing_env[:] = False
        if hasattr(twist_term, "is_heading_env"):
            twist_term.is_heading_env[:] = False

        t = state["step_idx"] * step_dt
        vx, vy, wz, label = _phase_at(t, total_schedule_s)
        twist_term.vel_command_b[:, 0] = vx
        twist_term.vel_command_b[:, 1] = vy
        twist_term.vel_command_b[:, 2] = wz

        if label != state["label"]:
            print(
                f"[t={t:7.2f}s] phase: {label:45s} "
                f"cmd=(vx={vx:+.2f} m/s, vy={vy:+.2f} m/s, wz={wz:+.2f} rad/s)"
            )
            state["label"] = label
        state["step_idx"] += 1

    twist_term.compute = scripted_compute  # type: ignore[assignment]

    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    print(f"[INFO] Viewer: {resolved_viewer}")
    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


if __name__ == "__main__":
    main()
