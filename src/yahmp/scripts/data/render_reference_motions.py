"""Render one reference video per motion clip.

This script does not run a policy. It loads a play environment, then
poses the robot directly on the reference motion and saves one `.mp4` per clip.
For directory-based motion sources, the output folder mirrors the
motion subfolder structure.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.tasks.registry import load_env_cfg

from yahmp.mdp import MotionCommand
from yahmp.mdp.motion.library import MotionLibrary


@dataclass(frozen=True)
class RenderReferenceMotionsConfig:
  """Configuration for batch rendering reference motions."""

  task_id: str = "Mjlab-YAHMP-Unitree-G1"
  motion_source: str | None = None
  output_dir: str | None = None
  device: str | None = None
  width: int | None = 1280
  height: int | None = 720
  fps: float | None = None
  max_motions: int | None = None
  overwrite: bool = False


def _resolve_motion_source(motion_cfg: Any, override: str | None) -> str:
  if override is not None:
    return override
  motion_source = getattr(motion_cfg, "motion_file", None)
  if not isinstance(motion_source, str) or motion_source == "":
    raise ValueError("Motion command config must define a non-empty `motion_file`.")
  return motion_source


def _resolve_relative_root(motion_source: str) -> Path:
  source = Path(motion_source).expanduser().resolve()
  if source.suffix in (".yaml", ".yml"):
    import yaml

    with open(source, "r", encoding="utf-8") as f:
      data = yaml.safe_load(f)
    if not isinstance(data, dict):
      raise ValueError(f"Motion YAML must be a mapping: {source}")
    root_path = data.get("root_path", ".")
    if not isinstance(root_path, str):
      raise ValueError(f"`root_path` must be a string in {source}.")
    resolved = Path(root_path)
    if not resolved.is_absolute():
      resolved = (source.parent / resolved).resolve()
    return resolved
  if source.is_dir():
    return source
  if source.suffix == ".npz":
    return source.parent
  raise ValueError(f"Unsupported motion source: {motion_source}")


def _default_output_dir(motion_source: str) -> Path:
  source = Path(motion_source).expanduser().resolve()
  if source.suffix in (".yaml", ".yml"):
    relative_root = _resolve_relative_root(motion_source)
    return relative_root.parent / f"{relative_root.name}_videos"
  if source.is_dir():
    return source.parent / f"{source.name}_videos"
  if source.suffix == ".npz":
    return source.parent / f"{source.stem}_videos"
  raise ValueError(f"Unsupported motion source: {motion_source}")


def _resolve_output_path(
  motion_file: Path, relative_root: Path, output_dir: Path
) -> Path:
  if motion_file.is_relative_to(relative_root):
    relative_path = motion_file.relative_to(relative_root)
    return (output_dir / relative_path).with_suffix(".mp4")
  return output_dir / f"{motion_file.stem}.mp4"


def _ensure_uint8(frame: np.ndarray) -> np.ndarray:
  if frame.dtype == np.uint8:
    return frame
  if np.issubdtype(frame.dtype, np.floating):
    return (np.clip(frame, 0.0, 1.0) * 255.0).astype(np.uint8)
  return np.clip(frame, 0, 255).astype(np.uint8)


def _write_video(path: Path, frames: list[np.ndarray], fps: float) -> None:
  import imageio.v2 as imageio

  path.parent.mkdir(parents=True, exist_ok=True)
  if len(frames) == 0:
    raise ValueError("Cannot write a video with no frames.")

  with imageio.get_writer(path, fps=fps) as writer:
    for frame in frames:
      writer.append_data(_ensure_uint8(frame))


def _frame_count_for_motion(length_s: float, fps: float) -> int:
  return max(int(math.ceil(length_s * fps)) + 1, 2)


def _build_motion_library_for_rendering(
  command: MotionCommand, motion_source: str
) -> tuple[MotionLibrary, int]:
  if command.motion_lib is not None:
    return command.motion_lib, command.motion_root_body_index

  required_body_names = tuple(
    dict.fromkeys(
      (command.cfg.anchor_body_name, *command.cfg.body_names, command.root_body_name)
    )
  )
  library = MotionLibrary(
    motion_source,
    device=command.device,
    show_progress=False,
    required_body_names=required_body_names,
  )
  return library, command.motion_root_body_index


def run(cfg: RenderReferenceMotionsConfig) -> None:
  """Render one video per motion clip."""
  import mjlab.tasks  # noqa: F401

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = 1
  if cfg.width is not None:
    env_cfg.viewer.width = cfg.width
  if cfg.height is not None:
    env_cfg.viewer.height = cfg.height

  motion_cfg = env_cfg.commands.get("motion")
  if motion_cfg is None:
    raise ValueError(f"Task does not define a `motion` command: {cfg.task_id}")

  motion_source = _resolve_motion_source(motion_cfg, cfg.motion_source)
  motion_cfg.motion_file = motion_source
  motion_cfg.debug_vis = False

  output_dir = (
    Path(cfg.output_dir).expanduser().resolve()
    if cfg.output_dir is not None
    else _default_output_dir(motion_source)
  )
  relative_root = _resolve_relative_root(motion_source)

  all_motion_files, _ = MotionLibrary._resolve_motion_entries(motion_source)
  motion_files = all_motion_files
  if cfg.max_motions is not None:
    motion_files = motion_files[: max(int(cfg.max_motions), 0)]

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode="rgb_array")
  env_ids = torch.tensor([0], dtype=torch.long, device=device)

  try:
    env.reset()

    command = cast(MotionCommand, env.command_manager.get_term("motion"))
    motion_lib, root_body_index = _build_motion_library_for_rendering(
      command, motion_source
    )
    robot = env.scene[command.cfg.entity_name]
    render_fps = float(cfg.fps if cfg.fps is not None else env.metadata["render_fps"])

    if len(all_motion_files) != motion_lib.num_motions():
      raise RuntimeError(
        "Resolved motion-file count does not match the loaded motion library."
      )

    env_origin = env.scene.env_origins[0:1]

    for motion_idx, motion_file in enumerate(motion_files):
      # Preserve the source-folder layout for outputs, including symlinked
      # selection directories, by computing the relative path before resolving.
      output_path = _resolve_output_path(
        motion_file=motion_file,
        relative_root=relative_root,
        output_dir=output_dir,
      )
      if output_path.exists() and not cfg.overwrite:
        print(f"[SKIP] {output_path}")
        continue

      length_s = float(motion_lib.motion_lengths_s[motion_idx].item())
      num_frames = _frame_count_for_motion(length_s, render_fps)
      max_time = max(length_s - 1.0e-6, 0.0)

      motion_ids = torch.full(
        (num_frames,), motion_idx, dtype=torch.long, device=command.device
      )
      motion_times = (
        torch.arange(num_frames, dtype=torch.float32, device=command.device)
        / render_fps
      )
      if max_time > 0.0:
        motion_times = torch.clamp(motion_times, max=max_time)
      else:
        motion_times.zero_()

      motion_frames = motion_lib.calc_motion_frame(
        motion_ids,
        motion_times,
        anchor_body_index=command.motion_anchor_body_index,
      )

      print(
        f"[INFO] Rendering {motion_idx + 1}/{len(motion_files)}: "
        f"{motion_file} -> {output_path}"
      )

      video_frames: list[np.ndarray] = []
      with torch.inference_mode():
        for frame_idx in range(num_frames):
          root_pos = motion_frames.body_pos_w[
            frame_idx : frame_idx + 1, root_body_index
          ]
          root_quat = motion_frames.body_quat_w[
            frame_idx : frame_idx + 1, root_body_index
          ]
          root_lin_vel = motion_frames.body_lin_vel_w[
            frame_idx : frame_idx + 1, root_body_index
          ]
          root_ang_vel = motion_frames.body_ang_vel_w[
            frame_idx : frame_idx + 1, root_body_index
          ]

          root_state = torch.cat(
            (
              root_pos + env_origin,
              root_quat,
              root_lin_vel,
              root_ang_vel,
            ),
            dim=-1,
          )
          joint_pos = motion_frames.joint_pos[frame_idx : frame_idx + 1]
          joint_vel = motion_frames.joint_vel[frame_idx : frame_idx + 1]

          robot.write_root_state_to_sim(root_state, env_ids=env_ids)
          robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
          robot.clear_state(env_ids=env_ids)
          env.sim.forward()

          frame = env.render()
          if frame is None:
            raise RuntimeError("Environment returned no frame while rendering.")
          frame_array = (
            frame[0] if isinstance(frame, np.ndarray) and frame.ndim == 4 else frame
          )
          video_frames.append(np.asarray(frame_array))

      _write_video(output_path, video_frames, render_fps)

    print(f"[INFO] Finished rendering videos to {output_dir}")
  finally:
    env.close()


if __name__ == "__main__":
  run(tyro.cli(RenderReferenceMotionsConfig))
