"""Compare YAHMP and Stiff-PD crouch logs for joint oscillations."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import numpy as np

os.environ["MPLCONFIGDIR"] = "/tmp/yahmp_matplotlib"

DEFAULT_YAHMP_LOG = (
  "paper/log/g1_motion_imitation/2026-06-21_12-17-22_YAHMP/"
  "0003_accad_A7___crouch.npz"
)
DEFAULT_STIFFPD_LOG = (
  "paper/log/g1_motion_imitation/2026-06-19_15-10-21_stiffPD/"
  "0001_accad_A7___crouch.npz"
)
DEFAULT_JOINTS = (
  "right_ankle_pitch_joint",
)


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--yahmp-log",
    type=Path,
    default=Path(DEFAULT_YAHMP_LOG),
    help="YAHMP real-robot crouch NPZ log.",
  )
  parser.add_argument(
    "--stiffpd-log",
    type=Path,
    default=Path(DEFAULT_STIFFPD_LOG),
    help="Stiff-PD real-robot crouch NPZ log.",
  )
  parser.add_argument(
    "--joints",
    nargs="+",
    default=list(DEFAULT_JOINTS),
    help="Joint names to compare.",
  )
  parser.add_argument(
    "--trend-window-s",
    type=float,
    default=0.40,
    help="Moving-average window used to remove the slow crouch trend.",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=Path("paper/log/g1_motion_imitation/stiffpd_vs_yahmp_crouch"),
    help="Directory for the plot and CSV summary.",
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
  hardware_names: list[str],
) -> np.ndarray:
  reference = np.asarray(_require(data, "reference_joint_pos"), dtype=np.float64)
  policy_names = [str(name) for name in _require(data, "meta_policy_joint_names")]
  if policy_names == hardware_names:
    return reference
  missing = sorted(set(hardware_names) - set(policy_names))
  if missing:
    raise ValueError(f"Reference is missing hardware joints: {missing}")
  indices = [policy_names.index(name) for name in hardware_names]
  return reference[:, indices]


def _moving_average(signal: np.ndarray, window_samples: int) -> np.ndarray:
  if window_samples <= 1:
    return signal.copy()
  if window_samples % 2 == 0:
    window_samples += 1
  pad = window_samples // 2
  padded = np.pad(signal, (pad, pad), mode="edge")
  kernel = np.ones(window_samples, dtype=np.float64) / float(window_samples)
  return np.convolve(padded, kernel, mode="valid")


def _load_log(path: Path) -> dict[str, object]:
  log_path = path.expanduser().resolve()
  if not log_path.is_file():
    raise FileNotFoundError(f"Log file not found: {log_path}")

  with np.load(log_path, allow_pickle=False) as data:
    if str(_scalar(data, "meta_episode_type")) != "motion_imitation":
      raise ValueError(f"Expected a `motion_imitation` log: {log_path}")
    mask = _motion_mask(data)
    if not np.any(mask):
      raise ValueError(f"No motion samples found in {log_path}")

    joint_names = [str(name) for name in _require(data, "meta_hardware_joint_names")]
    time_s = np.asarray(_require(data, "time_s"), dtype=np.float64)[mask]
    time_s = time_s - time_s[0]
    measured = np.asarray(_require(data, "joint_positions_hw"), dtype=np.float64)[mask]
    desired = np.asarray(_require(data, "desired_q_hw"), dtype=np.float64)[mask]
    reference = _reference_in_hardware_order(data, joint_names)[mask]
    torque = np.asarray(_require(data, "joint_torques_hw"), dtype=np.float64)[mask]

    return {
      "path": log_path,
      "policy": str(_scalar(data, "meta_onnx_name", log_path.parent.name)),
      "motion": str(_scalar(data, "meta_motion_name", "unknown")),
      "completed": bool(_scalar(data, "meta_completed", False)),
      "joint_names": joint_names,
      "time_s": time_s,
      "measured": measured,
      "desired": desired,
      "reference": reference,
      "torque": torque,
    }


def _joint_indices(joint_names: list[str], selected_joints: list[str]) -> list[int]:
  unknown = sorted(set(selected_joints) - set(joint_names))
  if unknown:
    raise ValueError(f"Unknown joint names: {unknown}")
  return [joint_names.index(name) for name in selected_joints]


def _display_joint_name(joint: str) -> str:
  labels = {
    "right_ankle_pitch_joint": "r. ankle pitch",
    "right_ankle_roll_joint": "r. ankle roll",
    "waist_yaw_joint": "waist yaw",
    "waist_roll_joint": "waist roll",
  }
  return labels.get(joint, joint.removesuffix("_joint"))


def _oscillation_stats(
  *,
  time_s: np.ndarray,
  measured: np.ndarray,
  reference: np.ndarray,
  desired: np.ndarray,
  torque: np.ndarray,
  trend_window_s: float,
) -> dict[str, float]:
  sample_dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else np.nan
  sample_rate = 1.0 / sample_dt if sample_dt > 0.0 else np.nan
  window_samples = max(3, int(round(trend_window_s * sample_rate)))
  tracking_error = measured - reference
  servo_error = measured - desired
  oscillation = tracking_error - _moving_average(tracking_error, window_samples)
  return {
    "tracking_rmse_rad": float(np.sqrt(np.mean(np.square(tracking_error)))),
    "servo_rmse_rad": float(np.sqrt(np.mean(np.square(servo_error)))),
    "oscillation_rms_rad": float(np.sqrt(np.mean(np.square(oscillation)))),
    "oscillation_p95_abs_rad": float(np.quantile(np.abs(oscillation), 0.95)),
    "oscillation_peak_to_peak_rad": float(np.max(oscillation) - np.min(oscillation)),
    "torque_rms_nm": float(np.sqrt(np.mean(np.square(torque)))),
    "torque_mean_abs_nm": float(np.mean(np.abs(torque))),
    "trend_window_samples": float(window_samples),
  }


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  fieldnames = [
    "policy",
    "joint",
    "tracking_rmse_rad",
    "servo_rmse_rad",
    "oscillation_rms_rad",
    "oscillation_p95_abs_rad",
    "oscillation_peak_to_peak_rad",
    "torque_rms_nm",
    "torque_mean_abs_nm",
    "trend_window_samples",
  ]
  with path.open("w", encoding="utf-8", newline="") as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


def _plot_comparison(
  *,
  output_path: Path,
  yahmp: dict[str, object],
  stiffpd: dict[str, object],
  joints: list[str],
) -> None:
  os.environ["MPLCONFIGDIR"] = "/tmp/yahmp_matplotlib"
  Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)
  import matplotlib.pyplot as plt

  yahmp_names = yahmp["joint_names"]
  stiffpd_names = stiffpd["joint_names"]
  assert isinstance(yahmp_names, list)
  assert isinstance(stiffpd_names, list)
  yahmp_indices = _joint_indices(yahmp_names, joints)
  stiffpd_indices = _joint_indices(stiffpd_names, joints)

  columns = min(2, len(joints))
  rows = int(np.ceil(len(joints) / columns))
  figure_size = (4.2, 1.75) if len(joints) == 1 else (7.2, 2.45 * rows)
  figure, axes = plt.subplots(
    rows,
    columns,
    figsize=figure_size,
    sharex=True,
    sharey=True,
    squeeze=False,
  )

  for plot_index, joint in enumerate(joints):
    row, col = divmod(plot_index, columns)
    axis = axes[row, col]
    yahmp_index = yahmp_indices[plot_index]
    stiffpd_index = stiffpd_indices[plot_index]
    axis.plot(
      yahmp["time_s"],
      yahmp["measured"][:, yahmp_index],
      linewidth=1.2,
      label="Mechanics-based",
    )
    axis.plot(
      stiffpd["time_s"],
      stiffpd["measured"][:, stiffpd_index],
      linewidth=1.2,
      label="Stiffer fixed-scale",
    )
    joint_label = _display_joint_name(joint)
    if len(joints) > 1:
      axis.set_title(joint_label, fontsize=9)
    if col == 0:
      if len(joints) == 1:
        axis.set_ylabel(f"{joint_label} [rad]")
      else:
        axis.set_ylabel("Measured angle [rad]")
    else:
      axis.tick_params(axis="y", labelleft=False)
    axis.grid(alpha=0.25)
    if row == rows - 1:
      axis.set_xlabel("Motion time [s]")

  for axis in axes.flat[len(joints) :]:
    axis.set_visible(False)

  if len(joints) == 1:
    axes.flat[0].legend(
      loc="upper right",
      ncol=1,
      frameon=False,
      fontsize=9,
      handlelength=1.8,
    )
    figure.subplots_adjust(left=0.22, right=0.995, bottom=0.24, top=0.97)
  else:
    handles, labels = axes.flat[0].get_legend_handles_labels()
    figure.legend(
      handles,
      labels,
      loc="upper center",
      ncol=2,
      frameon=False,
    )
    figure.subplots_adjust(
      left=0.085,
      right=0.995,
      bottom=0.16,
      top=0.84,
      wspace=0.04,
    )
  output_path.parent.mkdir(parents=True, exist_ok=True)
  figure.savefig(output_path.with_suffix(".pdf"), bbox_inches="tight")
  figure.savefig(output_path.with_suffix(".png"), dpi=220, bbox_inches="tight")
  plt.close(figure)


def run(args: argparse.Namespace) -> None:
  yahmp = _load_log(args.yahmp_log)
  stiffpd = _load_log(args.stiffpd_log)
  output_dir = args.output_dir.expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)

  rows: list[dict[str, object]] = []
  for policy_name, log in (("YAHMP", yahmp), ("Stiff-PD", stiffpd)):
    joint_names = log["joint_names"]
    assert isinstance(joint_names, list)
    indices = _joint_indices(joint_names, args.joints)
    for joint, index in zip(args.joints, indices, strict=True):
      stats = _oscillation_stats(
        time_s=log["time_s"],
        measured=log["measured"][:, index],
        reference=log["reference"][:, index],
        desired=log["desired"][:, index],
        torque=log["torque"][:, index],
        trend_window_s=args.trend_window_s,
      )
      rows.append({"policy": policy_name, "joint": joint, **stats})

  csv_path = output_dir / "stiffpd_vs_yahmp_crouch_oscillation_metrics.csv"
  figure_path = output_dir / "stiffpd_vs_yahmp_crouch_oscillations.png"
  _write_csv(csv_path, rows)
  _plot_comparison(
    output_path=figure_path,
    yahmp=yahmp,
    stiffpd=stiffpd,
    joints=list(args.joints),
  )

  print(f"YAHMP log     : {yahmp['path']}")
  print(f"Stiff-PD log  : {stiffpd['path']}")
  print(f"Output CSV    : {csv_path}")
  print(f"Output figure : {figure_path.with_suffix('.pdf')}")
  print(f"Output figure : {figure_path.with_suffix('.png')}")
  print()
  print("| Policy | Joint | osc. RMS [rad] | osc. P95 [rad] | torque RMS [N m] |")
  print("|---|---|---:|---:|---:|")
  for row in rows:
    print(
      "| "
      f"{row['policy']} | "
      f"{str(row['joint']).removesuffix('_joint')} | "
      f"{float(row['oscillation_rms_rad']):.4f} | "
      f"{float(row['oscillation_p95_abs_rad']):.4f} | "
      f"{float(row['torque_rms_nm']):.2f} |"
    )


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
