"""Plot one joint during the hold phase of a real-robot stance log."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np

DEFAULT_JOINT = "right_elbow_joint"


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("log_file", type=Path, help="Predefined-stance NPZ log.")
  parser.add_argument(
    "--joint",
    default=DEFAULT_JOINT,
    help=f"Joint to plot (default: {DEFAULT_JOINT}).",
  )
  parser.add_argument(
    "--output-dir",
    type=Path,
    default=None,
    help="Output directory (default: next to the input log).",
  )
  return parser.parse_args()


def _require(data: np.lib.npyio.NpzFile, key: str) -> np.ndarray:
  if key not in data:
    raise KeyError(f"Log is missing required array `{key}`.")
  return data[key]


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: object = "") -> object:
  return data[key].item() if key in data else default


def _hold_mask(data: np.lib.npyio.NpzFile) -> np.ndarray:
  phase_names = [str(name) for name in _require(data, "meta_phase_names")]
  if "hold" not in phase_names:
    raise ValueError(
      "This log has no `hold` phase. Available phases: " + ", ".join(phase_names)
    )
  return _require(data, "phase_id") == phase_names.index("hold")


def _reference_joint_index(
  data: np.lib.npyio.NpzFile,
  joint_name: str,
) -> int:
  policy_names = [str(name) for name in _require(data, "meta_policy_joint_names")]
  if joint_name not in policy_names:
    raise ValueError(f"Joint `{joint_name}` is not present in the policy joint list.")
  return policy_names.index(joint_name)


def _save_angle_plot(
  *,
  output_path: Path,
  time_s: np.ndarray,
  measured: np.ndarray,
  reference: np.ndarray,
  desired: np.ndarray,
  joint_name: str,
) -> None:
  import matplotlib.pyplot as plt

  figure, axis = plt.subplots(figsize=(10.0, 4.0))
  axis.plot(time_s, reference, "--", linewidth=1.4, label="Reference")
  axis.plot(time_s, desired, ":", linewidth=1.1, label="Commanded")
  axis.plot(time_s, measured, linewidth=1.1, label="Measured")
  axis.set_title(joint_name)
  axis.set_xlabel("Hold time [s]")
  axis.set_ylabel("Joint angle [rad]")
  axis.grid(alpha=0.25)
  axis.legend(frameon=False)
  figure.tight_layout()
  figure.savefig(output_path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def _save_torque_plot(
  *,
  output_path: Path,
  time_s: np.ndarray,
  torque: np.ndarray,
  joint_name: str,
) -> None:
  import matplotlib.pyplot as plt

  figure, axis = plt.subplots(figsize=(10.0, 4.0))
  axis.plot(time_s, torque, linewidth=1.0, color="#C44E52")
  axis.axhline(0.0, color="black", linewidth=0.7, alpha=0.5)
  axis.set_title(joint_name)
  axis.set_xlabel("Hold time [s]")
  axis.set_ylabel("Estimated joint torque [N m]")
  axis.grid(alpha=0.25)
  figure.tight_layout()
  figure.savefig(output_path, dpi=180, bbox_inches="tight")
  plt.close(figure)


def run(args: argparse.Namespace) -> None:
  os.environ.setdefault("MPLCONFIGDIR", "/tmp/yahmp_matplotlib")
  Path(os.environ["MPLCONFIGDIR"]).mkdir(parents=True, exist_ok=True)

  log_path = args.log_file.expanduser().resolve()
  if not log_path.is_file():
    raise FileNotFoundError(f"Log file not found: {log_path}")

  with np.load(log_path, allow_pickle=False) as data:
    if str(_scalar(data, "meta_episode_type")) != "predefined_stance":
      raise ValueError("Expected a `predefined_stance` log.")

    mask = _hold_mask(data)
    if not np.any(mask):
      raise ValueError("The log contains no samples in its `hold` phase.")

    hardware_names = [str(name) for name in _require(data, "meta_hardware_joint_names")]
    if args.joint not in hardware_names:
      raise ValueError(
        f"Unknown hardware joint `{args.joint}`. Available: {hardware_names}"
      )
    hardware_index = hardware_names.index(args.joint)
    reference_index = _reference_joint_index(data, args.joint)

    time_s = np.asarray(_require(data, "time_s"), dtype=np.float64)[mask]
    time_s = time_s - time_s[0]
    measured = np.asarray(_require(data, "joint_positions_hw"), dtype=np.float64)[
      mask, hardware_index
    ]
    reference = np.asarray(_require(data, "reference_joint_pos"), dtype=np.float64)[
      mask, reference_index
    ]
    desired = np.asarray(_require(data, "desired_q_hw"), dtype=np.float64)[
      mask, hardware_index
    ]
    torque = np.asarray(_require(data, "joint_torques_hw"), dtype=np.float64)[
      mask, hardware_index
    ]

    output_dir = (
      args.output_dir.expanduser().resolve()
      if args.output_dir is not None
      else log_path.parent
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    safe_joint_name = args.joint.removesuffix("_joint")
    angle_path = output_dir / f"{log_path.stem}_{safe_joint_name}_angle.png"
    torque_path = output_dir / f"{log_path.stem}_{safe_joint_name}_torque.png"

    _save_angle_plot(
      output_path=angle_path,
      time_s=time_s,
      measured=measured,
      reference=reference,
      desired=desired,
      joint_name=args.joint,
    )
    _save_torque_plot(
      output_path=torque_path,
      time_s=time_s,
      torque=torque,
      joint_name=args.joint,
    )

    sample_dt = float(np.median(np.diff(time_s))) if len(time_s) > 1 else np.nan
    sample_rate = 1.0 / sample_dt if sample_dt > 0.0 else np.nan
    print(f"Log                  : {log_path}")
    print(f"Policy               : {_scalar(data, 'meta_onnx_name', 'unknown')}")
    print(f"Pose                 : {_scalar(data, 'meta_pose_name', 'unknown')}")
    print(f"Joint                : {args.joint}")
    print(f"Hold samples         : {len(time_s)}")
    print(f"Hold duration        : {time_s[-1]:.3f} s")
    print(f"Measured sample rate : {sample_rate:.3f} Hz")
    print(f"Angle plot           : {angle_path}")
    print(f"Torque plot          : {torque_path}")


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
