"""Play an exported locomotion ONNX policy with a scripted velocity schedule.

This is the ONNX-runtime counterpart of scripts/play_scripted.py.  It keeps the
mjlab environment/viewer path intact, but replaces the PyTorch policy callable
with an ONNX Runtime policy callable.

Usage:
  uv run python scripts/play_onnx_scripted.py Mjlab-YAHMP-Locomotion-Unitree-G1 \\
      --onnx-file assets/models/multi_task/upper_posture\\=0_2.onnx \\
      --num-envs 1
"""

from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mjlab
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.utils.torch import configure_torch_backends
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

SCHEDULE: list[tuple[float, float, float, float, str]] = [
    (10.0, 0.0, 0.0, 0.0, "standing"),
    (6.0, 0.8, 0.0, 0.0, "walk medio (cmd noto nel training)"),
    (10.0, 1.0, 0.0, 0.0, "walk (vx=1.0)"),
    (8.0, 1.3, 0.0, 0.0, "walk (vx=1.5)"),
    (10.0, 2.0, 0.0, 0.0, "run (vx=2.0)"),
    (8.0, 2.0, 0.0, 0.5, "run turning (vx=2.5)"),
    (4.0, 1.5, 0.0, 0.5, "decel (vx=1.3)"),
    (2.0, 1.8, 0.0, 1.2, "decel (vx=1.3)"),
    (3.0, 0.0, -0.7, 0.0, "decel (vx=1.3)"),
    (2.0, 3.0, -1.0, 0.0, "decel (vx=1.3)"),
    (6.0, 3.0, 0.0, 0.0, "decel (vx=1.3)"),
    (4.0, -0.7, 0.0, 0.0, "decel (vx=-1.0)"),
    (4.0, -1.5, 0.0, 0.0, "decel (vx=-1.0)"),
    (6.0, 0.0, -0.7, 0.0, "lateral (v7=-1.0)"),
    # (6.0, 0.0, -1.5, 0.0, "lateral (v7=-1.0)"),
    (5.0, 0.0, 0.0, 1.0, "turn sul posto (wz=1.0)"),
    (6.0, 1.0, 0.0, 0.5, "walk + turn"),
    (4.0, -0.5, 0.0, 0.0, "walk indietro"),
]


@dataclass(frozen=True)
class Config:
    onnx_file: str
    """Path to the exported ONNX policy."""
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    ort_provider: Literal["auto", "cpu", "cuda"] = "auto"
    """ONNX Runtime execution provider preference."""
    inference_report_interval_s: float = 2.0
    """Seconds between average ONNX inference-time reports; <=0 disables."""
    no_terminations: bool = True
    """Disable terminations for a clean demo (the schedule keeps running)."""
    motion_file: str | None = None
    """Unused here, kept for parity with other play scripts."""


class OnnxPolicy:
    """Small adapter matching the mjlab viewer policy protocol."""

    def __init__(
        self,
        onnx_path: Path,
        *,
        device: str,
        provider: Literal["auto", "cpu", "cuda"],
        observation_group: str = "actor",
        report_interval_s: float = 2.0,
    ) -> None:
        try:
            import onnxruntime as ort
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "onnxruntime is required to run this script. Install/sync the "
                "project environment before using play_onnx_scripted.py."
            ) from exc

        available = ort.get_available_providers()
        if provider == "cpu":
            providers = ["CPUExecutionProvider"]
        elif provider == "cuda":
            if "CUDAExecutionProvider" not in available:
                raise RuntimeError(
                    "Requested CUDAExecutionProvider, but it is not available. "
                    f"Available providers: {available}"
                )
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = (
                ["CUDAExecutionProvider", "CPUExecutionProvider"]
                if "CUDAExecutionProvider" in available
                else ["CPUExecutionProvider"]
            )

        self.session = ort.InferenceSession(str(onnx_path), providers=providers)
        self.providers = providers
        self.input = self.session.get_inputs()[0]
        self.output = self.session.get_outputs()[0]
        self.device = torch.device(device)
        self.observation_group = self._metadata().get(
            "observation_group", observation_group
        )
        self.report_interval_s = float(report_interval_s)
        self._total_calls = 0
        self._total_inference_s = 0.0
        self._window_calls = 0
        self._window_inference_s = 0.0
        self._last_report_time = time.perf_counter()

    def __call__(self, obs: Any) -> torch.Tensor:
        if isinstance(obs, torch.Tensor):
            obs_tensor = obs
        else:
            obs_tensor = obs[self.observation_group]

        fixed_batch = _static_dim(self.input.shape, 0)
        if fixed_batch is not None and int(obs_tensor.shape[0]) != fixed_batch:
            raise ValueError(
                f"ONNX model expects batch size {fixed_batch}, but received "
                f"{int(obs_tensor.shape[0])}. Re-export with dynamic axes or run "
                f"with --num-envs {fixed_batch}."
            )

        expected_obs_dim = _static_dim(self.input.shape, 1)
        if (
            expected_obs_dim is not None
            and int(obs_tensor.shape[-1]) != expected_obs_dim
        ):
            raise ValueError(
                f"ONNX model expects observation dim {expected_obs_dim}, but "
                f"received {int(obs_tensor.shape[-1])}."
            )

        np_obs = (
            obs_tensor.detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .astype(np.float32, copy=False)
        )
        t0 = time.perf_counter()
        actions = self.session.run([self.output.name], {self.input.name: np_obs})[0]
        inference_s = time.perf_counter() - t0
        self._record_inference_time(inference_s)
        return torch.as_tensor(actions, dtype=torch.float32, device=self.device)

    def reset(self) -> None:
        """Policy reset hook used by the viewer; ONNX policy is stateless."""

    def _metadata(self) -> dict[str, str]:
        return dict(self.session.get_modelmeta().custom_metadata_map)

    def _record_inference_time(self, inference_s: float) -> None:
        self._total_calls += 1
        self._total_inference_s += inference_s
        self._window_calls += 1
        self._window_inference_s += inference_s

        if self.report_interval_s <= 0.0:
            return

        now = time.perf_counter()
        if now - self._last_report_time < self.report_interval_s:
            return

        window_avg_ms = 1000.0 * self._window_inference_s / self._window_calls
        total_avg_ms = 1000.0 * self._total_inference_s / self._total_calls
        print(
            f"[ONNX] inference avg: {window_avg_ms:.3f} ms "
            f"(last {self._window_calls} calls, total avg {total_avg_ms:.3f} ms)"
        )
        self._window_calls = 0
        self._window_inference_s = 0.0
        self._last_report_time = now


def _static_dim(shape: list[Any], idx: int) -> int | None:
    if idx >= len(shape):
        return None
    dim = shape[idx]
    return int(dim) if isinstance(dim, int) and dim > 0 else None


def _resolve_onnx_path(cfg: Config) -> Path:
    onnx_path = Path(cfg.onnx_file).expanduser().resolve()
    if not onnx_path.exists():
        raise FileNotFoundError(f"ONNX file not found: {onnx_path}")
    if onnx_path.suffix != ".onnx":
        raise ValueError(f"Expected an .onnx file, got: {onnx_path}")
    return onnx_path


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

    onnx_path = _resolve_onnx_path(cfg)
    print(f"[INFO] ONNX policy: {onnx_path}")

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

    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    policy = OnnxPolicy(
        onnx_path,
        device=device,
        provider=cfg.ort_provider,
        report_interval_s=cfg.inference_report_interval_s,
    )
    print(f"[INFO] ONNX Runtime providers: {policy.providers}")
    print(
        f"[INFO] ONNX IO: {policy.input.name}{policy.input.shape} -> "
        f"{policy.output.name}{policy.output.shape}"
    )

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
