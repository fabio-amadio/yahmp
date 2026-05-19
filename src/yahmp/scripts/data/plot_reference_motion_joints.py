"""View joint-by-joint position and velocity trajectories from a reference motion."""

from __future__ import annotations

import argparse
import os
import pickle
from pathlib import Path
from typing import Any

import numpy as np

os.environ.setdefault("MPLCONFIGDIR", "/tmp/yahmp_matplotlib")
Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)


G1_JOINT_ORDER = (
  "left_hip_pitch_joint",
  "left_hip_roll_joint",
  "left_hip_yaw_joint",
  "left_knee_joint",
  "left_ankle_pitch_joint",
  "left_ankle_roll_joint",
  "right_hip_pitch_joint",
  "right_hip_roll_joint",
  "right_hip_yaw_joint",
  "right_knee_joint",
  "right_ankle_pitch_joint",
  "right_ankle_roll_joint",
  "waist_yaw_joint",
  "waist_roll_joint",
  "waist_pitch_joint",
  "left_shoulder_pitch_joint",
  "left_shoulder_roll_joint",
  "left_shoulder_yaw_joint",
  "left_elbow_joint",
  "left_wrist_roll_joint",
  "left_wrist_pitch_joint",
  "left_wrist_yaw_joint",
  "right_shoulder_pitch_joint",
  "right_shoulder_roll_joint",
  "right_shoulder_yaw_joint",
  "right_elbow_joint",
  "right_wrist_roll_joint",
  "right_wrist_pitch_joint",
  "right_wrist_yaw_joint",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "motion_file",
    type=Path,
    help="Reference motion file to plot. Supports YAHMP .npz and TWIST2-style .pkl.",
  )
  parser.add_argument(
    "--output",
    type=Path,
    default=None,
    help=(
      "Optional output path. If multiple joints are selected, each joint is saved "
      "as <stem>_<joint-name><suffix>. By default nothing is saved."
    ),
  )
  parser.add_argument(
    "--joints",
    nargs="+",
    default=("all",),
    help=(
      "Joint names or zero-based indices to plot. Use `all` to plot every joint. "
      "G1 joint names are available when the motion has 29 DoFs."
    ),
  )
  parser.add_argument(
    "--start",
    type=float,
    default=0.0,
    help="Start time in seconds.",
  )
  parser.add_argument(
    "--end",
    type=float,
    default=None,
    help="End time in seconds. Defaults to the end of the clip.",
  )
  parser.add_argument(
    "--dpi",
    type=int,
    default=160,
    help="Saved figure DPI.",
  )
  parser.add_argument(
    "--show",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Show plots interactively. Use --no-show for save-only batch usage.",
  )
  return parser.parse_args()


def _as_float_array(data: Any, key: str, source: Path) -> np.ndarray:
  if key not in data:
    raise KeyError(f"Motion file is missing `{key}`: {source}")
  return np.asarray(data[key], dtype=np.float32)


def _extract_scalar_fps(data: Any, default: float = 30.0) -> float:
  if "fps" not in data:
    return default
  fps_values = np.asarray(data["fps"], dtype=np.float64).reshape(-1)
  if fps_values.size == 0 or fps_values[0] <= 1.0e-6:
    return default
  return float(fps_values[0])


def _finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
  if values.shape[0] < 2:
    return np.zeros_like(values, dtype=np.float32)
  return np.gradient(values, 1.0 / fps, axis=0).astype(np.float32)


def _load_npz_motion(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
  with np.load(path, allow_pickle=False) as data:
    joint_pos = _as_float_array(data, "joint_pos", path)
    joint_vel = (
      np.asarray(data["joint_vel"], dtype=np.float32) if "joint_vel" in data else None
    )
    fps = _extract_scalar_fps(data)
  if joint_vel is None:
    joint_vel = _finite_difference(joint_pos, fps)
  return fps, joint_pos, joint_vel


def _load_pkl_motion(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
  with path.open("rb") as f:
    data = pickle.load(f)
  if not isinstance(data, dict):
    raise ValueError(f"Expected a dict in motion pickle: {path}")

  joint_pos = _as_float_array(data, "dof_pos", path)
  joint_vel = (
    np.asarray(data["joint_vel"], dtype=np.float32) if "joint_vel" in data else None
  )
  fps = _extract_scalar_fps(data)
  if joint_vel is None:
    joint_vel = _finite_difference(joint_pos, fps)
  return fps, joint_pos, joint_vel


def load_joint_motion(path: Path) -> tuple[float, np.ndarray, np.ndarray]:
  path = path.expanduser()
  if not path.exists():
    raise FileNotFoundError(f"Motion file not found: {path}")
  if path.suffix == ".npz":
    fps, joint_pos, joint_vel = _load_npz_motion(path)
  elif path.suffix in (".pkl", ".pickle"):
    fps, joint_pos, joint_vel = _load_pkl_motion(path)
  else:
    raise ValueError(f"Unsupported motion file extension: {path.suffix}")

  if joint_pos.ndim != 2:
    raise ValueError(
      f"Expected joint_pos/dof_pos shape [frames, joints], got {joint_pos.shape}"
    )
  if joint_vel.shape != joint_pos.shape:
    raise ValueError(
      f"joint_vel shape must match joint positions: {joint_vel.shape} vs {joint_pos.shape}"
    )
  return fps, joint_pos, joint_vel


def joint_names_for_count(num_joints: int) -> tuple[str, ...]:
  if num_joints == len(G1_JOINT_ORDER):
    return G1_JOINT_ORDER
  return tuple(f"joint_{i:02d}" for i in range(num_joints))


def resolve_joint_indices(
  selectors: tuple[str, ...],
  joint_names: tuple[str, ...],
) -> list[int]:
  if len(selectors) == 0 or any(selector.lower() == "all" for selector in selectors):
    return list(range(len(joint_names)))

  name_to_index = {name: idx for idx, name in enumerate(joint_names)}
  indices: list[int] = []
  for selector in selectors:
    try:
      index = int(selector)
    except ValueError:
      if selector not in name_to_index:
        raise ValueError(
          f"Unknown joint `{selector}`. Available names: {', '.join(joint_names)}"
        ) from None
      index = name_to_index[selector]

    if index < 0 or index >= len(joint_names):
      raise IndexError(
        f"Joint index {index} is out of range for {len(joint_names)} joints."
      )
    if index not in indices:
      indices.append(index)
  return indices


def slice_time_window(
  fps: float,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  start_s: float,
  end_s: float | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  if start_s < 0.0:
    raise ValueError("--start must be non-negative.")
  if end_s is not None and end_s <= start_s:
    raise ValueError("--end must be greater than --start.")

  frame_count = joint_pos.shape[0]
  start_frame = min(int(np.floor(start_s * fps)), frame_count - 1)
  end_frame = (
    frame_count if end_s is None else min(int(np.ceil(end_s * fps)) + 1, frame_count)
  )
  if end_frame <= start_frame:
    raise ValueError("Selected time window contains no frames.")

  frames = np.arange(start_frame, end_frame)
  times = frames.astype(np.float64) / fps
  return times, joint_pos[start_frame:end_frame], joint_vel[start_frame:end_frame]


def _safe_filename(name: str) -> str:
  safe_chars = [char if char.isalnum() or char in ("-", "_") else "_" for char in name]
  return "".join(safe_chars).strip("_") or "joint"


def _joint_output_path(base_output: Path, joint_name: str, multi_joint: bool) -> Path:
  if not multi_joint:
    return base_output
  return base_output.with_name(
    f"{base_output.stem}_{_safe_filename(joint_name)}{base_output.suffix}"
  )


def _plot_joint(
  figure: Any,
  axes: tuple[Any, Any],
  times: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  joint_index: int,
  joint_name: str,
  motion_file: Path,
  fps: float,
) -> None:
  pos_ax, vel_ax = axes
  pos_ax.plot(times, joint_pos[:, joint_index], linewidth=1.25, color="#1f77b4")
  vel_ax.plot(times, joint_vel[:, joint_index], linewidth=1.25, color="#d62728")

  figure.suptitle(
    f"{joint_name} | {motion_file.name} | {joint_pos.shape[0]} frames at {fps:.3g} fps"
  )
  pos_ax.set_ylabel("Position [rad]")
  vel_ax.set_ylabel("Velocity [rad/s]")
  vel_ax.set_xlabel("Time [s]")
  pos_ax.grid(True, alpha=0.25)
  vel_ax.grid(True, alpha=0.25)
  pos_ax.set_title("Position")
  vel_ax.set_title("Velocity")


def plot_motion(
  motion_file: Path,
  output_path: Path | None,
  fps: float,
  times: np.ndarray,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray,
  indices: list[int],
  joint_names: tuple[str, ...],
  dpi: int,
  show: bool,
) -> None:
  import matplotlib.pyplot as plt

  figures: list[Any] = []
  for index in indices:
    joint_name = joint_names[index]
    figure, axes = plt.subplots(2, 1, figsize=(11.0, 6.5), sharex=True)
    assert len(axes) == 2
    _plot_joint(
      figure,
      (axes[0], axes[1]),
      times,
      joint_pos,
      joint_vel,
      index,
      joint_name,
      motion_file,
      fps,
    )
    figure.tight_layout()
    figures.append(figure)

    if output_path is not None:
      joint_output = _joint_output_path(output_path, joint_name, len(indices) > 1)
      joint_output.parent.mkdir(parents=True, exist_ok=True)
      figure.savefig(joint_output, dpi=dpi)
      print(f"[INFO] Wrote {joint_output}")

  if show:
    plt.show()
  for figure in figures:
    plt.close(figure)


def main() -> None:
  args = parse_args()
  fps, joint_pos, joint_vel = load_joint_motion(args.motion_file)
  joint_names = joint_names_for_count(joint_pos.shape[1])
  indices = resolve_joint_indices(tuple(args.joints), joint_names)
  times, window_pos, window_vel = slice_time_window(
    fps,
    joint_pos,
    joint_vel,
    args.start,
    args.end,
  )
  output_path = args.output
  plot_motion(
    motion_file=args.motion_file,
    output_path=output_path,
    fps=fps,
    times=times,
    joint_pos=window_pos,
    joint_vel=window_vel,
    indices=indices,
    joint_names=joint_names,
    dpi=args.dpi,
    show=args.show,
  )


if __name__ == "__main__":
  main()
