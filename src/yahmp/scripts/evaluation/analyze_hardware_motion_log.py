"""Analyze the motion phase of a YAHMP real-robot motion-imitation log."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("log_file", type=Path, help="Motion-imitation NPZ log.")
  parser.add_argument(
    "--plot-joint-angles",
    action="store_true",
    help="Save measured, reference, and commanded joint-angle trajectories.",
  )
  parser.add_argument(
    "--plot-output",
    type=Path,
    default=None,
    help="Joint-angle plot path (default: next to the input log).",
  )
  parser.add_argument(
    "--joints",
    nargs="*",
    default=None,
    help="Joint names to plot. By default, all joints are plotted.",
  )
  return parser.parse_args()


def _require(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
  if key not in data:
    raise KeyError(f"Log is missing required array `{key}`.")
  return data[key]


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: object = "") -> object:
  return data[key].item() if key in data else default


def _motion_mask(data: np.lib.npyio.NpzFile) -> np.ndarray:
  phase_names = [str(name) for name in _require(data, "meta_phase_names")]
  if "motion" not in phase_names:
    raise ValueError(
      "This log has no `motion` phase. Available phases: " + ", ".join(phase_names)
    )
  return _require(data, "phase_id") == phase_names.index("motion")


def _reference_in_hardware_order(
  data: np.lib.npyio.NpzFile,
  key: str,
  hardware_names: list[str],
) -> np.ndarray:
  values = np.asarray(_require(data, key), dtype=np.float64)
  policy_names = [str(name) for name in _require(data, "meta_policy_joint_names")]
  if policy_names == hardware_names:
    return values
  missing = sorted(set(hardware_names) - set(policy_names))
  if missing:
    raise ValueError(f"Reference is missing hardware joints: {missing}")
  indices = [policy_names.index(name) for name in hardware_names]
  return values[:, indices]


def _tracking_stats(error: np.ndarray) -> dict[str, float]:
  return {
    "mean_l2": float(np.mean(np.linalg.norm(error, axis=1))),
    "rmse": float(np.sqrt(np.mean(np.square(error)))),
    "mae": float(np.mean(np.abs(error))),
    "p95_abs": float(np.quantile(np.abs(error), 0.95)),
    "max_abs": float(np.max(np.abs(error))),
  }


def _torque_stats(torque: np.ndarray) -> dict[str, float]:
  absolute = np.abs(torque)
  return {
    "mean_abs": float(np.mean(absolute)),
    "rms": float(np.sqrt(np.mean(np.square(torque)))),
    "p95_abs": float(np.quantile(absolute, 0.95)),
    "max_abs": float(np.max(absolute)),
  }


def _print_tracking(title: str, stats: dict[str, float], unit: str) -> None:
  print(title)
  print(f"  Mean per-sample L2 norm : {stats['mean_l2']:.6f} {unit}")
  print(f"  Per-joint RMSE         : {stats['rmse']:.6f} {unit}")
  print(f"  Per-joint MAE          : {stats['mae']:.6f} {unit}")
  print(f"  Per-joint P95 abs.     : {stats['p95_abs']:.6f} {unit}")
  print(f"  Per-joint max abs.     : {stats['max_abs']:.6f} {unit}")


def _print_torque(title: str, stats: dict[str, float]) -> None:
  print(title)
  print(f"  Mean absolute torque   : {stats['mean_abs']:.6f} N m")
  print(f"  RMS torque             : {stats['rms']:.6f} N m")
  print(f"  P95 absolute torque    : {stats['p95_abs']:.6f} N m")
  print(f"  Maximum absolute torque: {stats['max_abs']:.6f} N m")


def _plot_joint_angles(
  *,
  output_path: Path,
  time_s: np.ndarray,
  measured: np.ndarray,
  reference: np.ndarray,
  desired: np.ndarray,
  joint_names: list[str],
  selected_names: list[str] | None,
) -> None:
  os.environ.setdefault("MPLCONFIGDIR", "/tmp/yahmp_matplotlib")
  Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
  import matplotlib.pyplot as plt

  if selected_names:
    unknown = sorted(set(selected_names) - set(joint_names))
    if unknown:
      raise ValueError(f"Unknown joint names: {unknown}")
    indices = [joint_names.index(name) for name in selected_names]
  else:
    indices = list(range(len(joint_names)))

  columns = min(3, len(indices))
  rows = int(np.ceil(len(indices) / columns))
  figure, axes = plt.subplots(
    rows,
    columns,
    figsize=(5.2 * columns, 2.25 * rows),
    sharex=True,
    squeeze=False,
  )
  relative_time = time_s - time_s[0]
  for axis, joint_index in zip(axes.flat, indices, strict=False):
    axis.plot(
      relative_time, reference[:, joint_index], "--", linewidth=1.1, label="Reference"
    )
    axis.plot(
      relative_time, desired[:, joint_index], ":", linewidth=1.0, label="Commanded"
    )
    axis.plot(relative_time, measured[:, joint_index], linewidth=1.0, label="Measured")
    axis.set_title(joint_names[joint_index].removesuffix("_joint"), fontsize=9)
    axis.set_ylabel("Angle [rad]")
    axis.grid(alpha=0.25)
  for axis in axes.flat[len(indices) :]:
    axis.set_visible(False)
  for axis in axes[-1, :]:
    if axis.get_visible():
      axis.set_xlabel("Motion time [s]")
  handles, labels = axes.flat[0].get_legend_handles_labels()
  figure.legend(handles, labels, loc="upper center", ncol=3, frameon=False)
  figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.98))
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def run(args: argparse.Namespace) -> None:
  log_path = args.log_file.expanduser().resolve()
  if not log_path.is_file():
    raise FileNotFoundError(f"Log file not found: {log_path}")

  with np.load(log_path, allow_pickle=False) as data:
    if str(_scalar(data, "meta_episode_type")) != "motion_imitation":
      raise ValueError("Expected a `motion_imitation` log.")

    mask = _motion_mask(data)
    if not np.any(mask):
      raise ValueError("The log contains no samples in its `motion` phase.")

    joint_names = [str(name) for name in _require(data, "meta_hardware_joint_names")]
    measured_pos = np.asarray(_require(data, "joint_positions_hw"), dtype=np.float64)[
      mask
    ]
    measured_vel = np.asarray(_require(data, "joint_velocities_hw"), dtype=np.float64)[
      mask
    ]
    measured_torque = np.asarray(_require(data, "joint_torques_hw"), dtype=np.float64)[
      mask
    ]
    desired_pos = np.asarray(_require(data, "desired_q_hw"), dtype=np.float64)[mask]
    reference_pos = _reference_in_hardware_order(
      data, "reference_joint_pos", joint_names
    )[mask]
    reference_vel = _reference_in_hardware_order(
      data, "reference_joint_vel", joint_names
    )[mask]
    time_s = np.asarray(_require(data, "time_s"), dtype=np.float64)[mask]

    lower_indices = np.asarray(
      [
        index
        for index, name in enumerate(joint_names)
        if any(part in name for part in ("hip", "knee", "ankle"))
      ],
      dtype=np.int64,
    )
    upper_indices = np.asarray(
      [index for index in range(len(joint_names)) if index not in set(lower_indices)],
      dtype=np.int64,
    )

    duration = float(time_s[-1] - time_s[0]) if len(time_s) > 1 else 0.0
    sample_dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else np.nan
    sample_rate = 1.0 / sample_dt if sample_dt > 0.0 else np.nan

    print(f"Log                    : {log_path}")
    print(f"Policy                 : {_scalar(data, 'meta_onnx_name', 'unknown')}")
    print(f"Motion                 : {_scalar(data, 'meta_motion_name', 'unknown')}")
    print(f"Completed              : {_scalar(data, 'meta_completed', False)}")
    print(f"Motion samples         : {len(time_s)}")
    print(f"Motion duration        : {duration:.3f} s")
    print(f"Measured sample rate   : {sample_rate:.3f} Hz")
    print()

    _print_tracking(
      "Joint-position tracking (measured - motion reference)",
      _tracking_stats(measured_pos - reference_pos),
      "rad",
    )
    print()
    _print_tracking(
      "Joint-velocity tracking (measured - motion reference)",
      _tracking_stats(measured_vel - reference_vel),
      "rad/s",
    )
    print()
    _print_tracking(
      "Servo position tracking (measured - commanded target)",
      _tracking_stats(measured_pos - desired_pos),
      "rad",
    )
    print()
    _print_torque("All joints", _torque_stats(measured_torque))
    print()
    _print_torque("Lower body", _torque_stats(measured_torque[:, lower_indices]))
    print()
    _print_torque("Upper body", _torque_stats(measured_torque[:, upper_indices]))

    if args.plot_joint_angles:
      output_path = (
        args.plot_output.expanduser().resolve()
        if args.plot_output is not None
        else log_path.with_name(f"{log_path.stem}_joint_angles.png")
      )
      _plot_joint_angles(
        output_path=output_path,
        time_s=time_s,
        measured=measured_pos,
        reference=reference_pos,
        desired=desired_pos,
        joint_names=joint_names,
        selected_names=args.joints,
      )
      print()
      print(f"Joint-angle plot        : {output_path}")


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
