"""Play one locomotion checkpoint on a scripted forward-velocity ramp.

The "twist" command term is overwritten at every env step with a simple
schedule: stand, linearly increase forward velocity, then hold the final speed.

Example:
  uv run python scripts/play_quantizer_ramp.py Mjlab-YAHMP-Locomotion-Unitree-G1 \
      --checkpoint-file /home/famadio/Workspace/2026-06-24_10-05-40_Loco-Quant-5/model_4999.pt \
      --imitation-checkpoint-file assets/models/multi_task/imitation_rvq_model.pt \
      --rvq-num-active-quantizers 5
"""

from __future__ import annotations

import math
import os
import re
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
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


@dataclass(frozen=True)
class Config:
    checkpoint_file: str
    """Path to the trained locomotion checkpoint."""

    imitation_checkpoint_file: str | None = None
    """Frozen imitation/RVQ checkpoint required by the YAHMP locomotion runner."""

    rvq_num_active_quantizers: int | None = None
    """Active RVQ quantizers. If omitted, tries to infer from the checkpoint path."""

    start_vx: float = 0.3
    end_vx: float = 3.0
    stand_s: float = 0.0
    ramp_s: float = 20.0
    hold_s: float = 10.0
    """Forward command schedule: stand, linearly ramp vx, then hold end_vx."""

    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    duration_s: float | None = None
    """Viewer duration. None runs until the viewer is closed."""

    video: bool = False
    """Run headless and save an mp4 instead of opening a viewer."""

    video_length_s: float | None = None
    """Video duration. None uses duration_s, or stand_s + ramp_s + hold_s."""

    video_dir: str = "outputs/ramp_videos"
    """Directory where headless videos are written."""

    video_height: int | None = None
    video_width: int | None = None
    """Optional offscreen render resolution."""

    azimuth: float | None = None
    elevation: float | None = None
    distance: float | None = None
    """Optional camera overrides in degrees/meters."""

    no_terminations: bool = True
    """Disable terminations for a clean demo."""


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


def _infer_quantizers_from_path(checkpoint_path: Path) -> int | None:
    text = str(checkpoint_path)
    match = re.search(r"(?:Quant-|quant_)(\d+)", text)
    return int(match.group(1)) if match else None


def _set_runner_registry_options(
    task_name: str,
    *,
    imitation_checkpoint_file: str | None,
    rvq_num_active_quantizers: int | None,
) -> None:
    registered_cfg = _REGISTRY[task_name].rl_cfg
    if imitation_checkpoint_file is not None:
        if not hasattr(registered_cfg, "imitation_checkpoint_file"):
            raise ValueError(
                f"Task {task_name} runner cfg does not accept `imitation_checkpoint_file`."
            )
        registered_cfg.imitation_checkpoint_file = imitation_checkpoint_file

    if rvq_num_active_quantizers is not None:
        if rvq_num_active_quantizers <= 0:
            raise ValueError("rvq_num_active_quantizers must be positive.")
        if not hasattr(registered_cfg, "rvq_num_active_quantizers"):
            raise ValueError(
                f"Task {task_name} runner cfg does not accept `rvq_num_active_quantizers`."
            )
        registered_cfg.rvq_num_active_quantizers = rvq_num_active_quantizers


def _configure_demo_env(env_cfg, *, episode_length_s: float, no_terminations: bool) -> None:
    env_cfg.scene.num_envs = 1
    env_cfg.episode_length_s = episode_length_s

    reset_base = env_cfg.events.get("reset_base")
    if reset_base is not None:
        reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        reset_base.params["velocity_range"] = {}
    reset_joints = env_cfg.events.get("reset_robot_joints")
    if reset_joints is not None:
        reset_joints.params["position_range"] = (0.0, 0.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)

    if no_terminations:
        env_cfg.terminations = {}


def _configure_camera(env_cfg, cfg: Config) -> None:
    if cfg.azimuth is not None:
        env_cfg.viewer.azimuth = cfg.azimuth
    if cfg.elevation is not None:
        env_cfg.viewer.elevation = cfg.elevation
    if cfg.distance is not None:
        env_cfg.viewer.distance = cfg.distance


def _ramp_command(t: float, cfg: Config) -> tuple[float, str]:
    if t < cfg.stand_s:
        return 0.0, "standing"
    ramp_t = min(max((t - cfg.stand_s) / max(cfg.ramp_s, 1e-6), 0.0), 1.0)
    vx = cfg.start_vx + ramp_t * (cfg.end_vx - cfg.start_vx)
    if ramp_t < 1.0:
        return vx, "ramping"
    return cfg.end_vx, "holding"


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

    rvq_num_active_quantizers = (
        cfg.rvq_num_active_quantizers or _infer_quantizers_from_path(checkpoint_path)
    )
    _set_runner_registry_options(
        chosen_task,
        imitation_checkpoint_file=cfg.imitation_checkpoint_file,
        rvq_num_active_quantizers=rvq_num_active_quantizers,
    )

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    demo_length_s = cfg.stand_s + cfg.ramp_s + cfg.hold_s
    episode_length_s = max(cfg.duration_s or demo_length_s + 60.0, demo_length_s + 2.0)

    env_cfg = load_env_cfg(chosen_task, play=True)
    agent_cfg = load_rl_cfg(chosen_task)
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width
    _configure_camera(env_cfg, cfg)
    _configure_demo_env(
        env_cfg,
        episode_length_s=episode_length_s,
        no_terminations=cfg.no_terminations,
    )

    render_mode = "rgb_array" if cfg.video else None
    raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)
    video_steps: int | None = None
    video_path: Path | None = None
    if cfg.video:
        video_length_s = (
            cfg.video_length_s
            if cfg.video_length_s is not None
            else (cfg.duration_s if cfg.duration_s is not None else demo_length_s)
        )
        if video_length_s <= 0.0:
            raise ValueError("video_length_s must be positive.")
        video_steps = int(math.ceil(video_length_s / raw_env.step_dt))
        video_dir = Path(cfg.video_dir).expanduser().resolve()
        video_path = video_dir / "ramp-step-0.mp4"
        raw_env = VideoRecorder(
            raw_env,
            video_folder=video_dir,
            step_trigger=lambda step: step == 0,
            video_length=video_steps,
            name_prefix="ramp",
            disable_logger=False,
        )

    env = raw_env
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    actor = getattr(runner.alg, "actor", None)
    resolved_quantizers = getattr(actor, "num_active_codebooks", None)
    if rvq_num_active_quantizers is not None and resolved_quantizers is not None:
        if int(resolved_quantizers) != int(rvq_num_active_quantizers):
            raise RuntimeError(
                "Active RVQ quantizer mismatch: requested "
                f"{rvq_num_active_quantizers}, actor resolved {resolved_quantizers}."
            )

    runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    cmd_mgr = env.unwrapped.command_manager
    twist_name = _find_twist_term_name(cmd_mgr.active_terms)
    twist_term = cmd_mgr.get_term(twist_name)
    if hasattr(twist_term, "cfg"):
        if hasattr(twist_term.cfg, "rel_standing_envs"):
            twist_term.cfg.rel_standing_envs = 0.0
        if hasattr(twist_term.cfg, "rel_heading_envs"):
            twist_term.cfg.rel_heading_envs = 0.0
        if hasattr(twist_term.cfg, "init_velocity_prob"):
            twist_term.cfg.init_velocity_prob = 0.0
    if hasattr(twist_term, "_joystick_enabled"):
        twist_term._joystick_enabled = None

    step_dt = float(env.unwrapped.step_dt)
    state = {"step_idx": 0, "label": None, "last_print_second": -1}
    original_compute = twist_term.compute

    def ramp_compute(dt: float) -> None:
        original_compute(dt)
        if hasattr(twist_term, "is_standing_env"):
            twist_term.is_standing_env[:] = False
        if hasattr(twist_term, "is_heading_env"):
            twist_term.is_heading_env[:] = False

        t = state["step_idx"] * step_dt
        vx, label = _ramp_command(t, cfg)
        twist_term.vel_command_b[:, 0] = vx
        twist_term.vel_command_b[:, 1] = 0.0
        twist_term.vel_command_b[:, 2] = 0.0

        whole_second = int(math.floor(t))
        should_print = label != state["label"] or (
            whole_second % 5 == 0 and whole_second != state["last_print_second"]
        )
        if should_print:
            print(f"[t={t:7.2f}s] {label:8s} cmd=(vx={vx:+.2f}, vy=+0.00, wz=+0.00)")
            state["label"] = label
            state["last_print_second"] = whole_second
        state["step_idx"] += 1

    twist_term.compute = ramp_compute  # type: ignore[assignment]

    if cfg.viewer == "auto":
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    print(f"[INFO] Task: {chosen_task}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {device}")
    if resolved_quantizers is not None:
        print(f"[INFO] Active RVQ quantizers: {resolved_quantizers}")
    print(
        "[INFO] Ramp: "
        f"stand {cfg.stand_s:.1f}s, vx {cfg.start_vx:.2f}->{cfg.end_vx:.2f} "
        f"over {cfg.ramp_s:.1f}s, hold {cfg.hold_s:.1f}s"
    )
    if cfg.video:
        assert video_steps is not None
        print(f"[INFO] Video: {video_path} ({video_steps} steps)")
        try:
            for _ in range(video_steps):
                obs = env.get_observations()
                with torch.inference_mode():
                    actions = policy(obs)
                env.step(actions)
        finally:
            env.close()
        return

    print(f"[INFO] Viewer: {resolved_viewer}")

    num_steps = None if cfg.duration_s is None else int(math.ceil(cfg.duration_s / step_dt))
    try:
        if resolved_viewer == "native":
            NativeMujocoViewer(env, policy).run(num_steps=num_steps)
        elif resolved_viewer == "viser":
            ViserPlayViewer(env, policy).run(num_steps=num_steps)
        else:
            raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")
    finally:
        env.close()


if __name__ == "__main__":
    main()
