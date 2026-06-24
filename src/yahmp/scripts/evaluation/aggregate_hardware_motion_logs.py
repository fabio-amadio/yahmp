"""Aggregate completed real-robot motion-imitation logs."""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import mujoco
import numpy as np
from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

from yahmp.config.g1.env_cfgs import G1_COMPARISON_KEY_BODY_NAMES
from yahmp.scripts.evaluation.analyze_hardware_motion_log import (
  _motion_mask,
  _reference_in_hardware_order,
  _torque_stats,
  _tracking_stats,
)

METRIC_UNITS = {
  "key_body_pos_mean": "m",
  "key_body_pos_rmse": "m",
  "key_body_pos_p95": "m",
  "key_body_pos_max": "m",
  "key_body_rot_mean": "rad",
  "key_body_rot_rmse": "rad",
  "key_body_rot_p95": "rad",
  "key_body_rot_max": "rad",
  "joint_pos_mean_l2": "rad",
  "joint_pos_rmse": "rad",
  "joint_pos_mae": "rad",
  "joint_pos_p95_abs": "rad",
  "joint_pos_max_abs": "rad",
  "joint_vel_mean_l2": "rad/s",
  "joint_vel_rmse": "rad/s",
  "joint_vel_mae": "rad/s",
  "joint_vel_p95_abs": "rad/s",
  "joint_vel_max_abs": "rad/s",
  "servo_pos_mean_l2": "rad",
  "servo_pos_rmse": "rad",
  "servo_pos_mae": "rad",
  "servo_pos_p95_abs": "rad",
  "servo_pos_max_abs": "rad",
  "torque_mean_abs": "N m",
  "torque_rms": "N m",
  "torque_p95_abs": "N m",
  "torque_max_abs": "N m",
  "lower_torque_mean_abs": "N m",
  "lower_torque_rms": "N m",
  "lower_torque_p95_abs": "N m",
  "lower_torque_max_abs": "N m",
  "upper_torque_mean_abs": "N m",
  "upper_torque_rms": "N m",
  "upper_torque_p95_abs": "N m",
  "upper_torque_max_abs": "N m",
}


def parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "log_dir",
    nargs="?",
    type=Path,
    default=Path("paper/log/g1_motion_imitation"),
    help="Directory searched recursively for NPZ logs.",
  )
  parser.add_argument(
    "--csv",
    type=Path,
    default=None,
    help="Optionally save one row of metrics per included execution.",
  )
  return parser.parse_args()


def _scalar(data: np.lib.npyio.NpzFile, key: str, default: object = "") -> object:
  return data[key].item() if key in data else default


def _prefixed(prefix: str, values: dict[str, float]) -> dict[str, float]:
  return {f"{prefix}_{key}": value for key, value in values.items()}


def _scalar_error_stats(error: np.ndarray) -> dict[str, float]:
  return {
    "mean": float(np.mean(error)),
    "rmse": float(np.sqrt(np.mean(np.square(error)))),
    "p95": float(np.quantile(error, 0.95)),
    "max": float(np.max(error)),
  }


class G1ForwardKinematics:
  """Compute key-body poses relative to the pelvis from G1 joint positions."""

  def __init__(self, joint_names: list[str]):
    self.joint_names = tuple(joint_names)
    self.model = mujoco.MjModel.from_xml_path(str(G1_XML))
    self.data = mujoco.MjData(self.model)

    self.pelvis_body_id = self._body_id("pelvis")
    self.key_body_ids = np.asarray(
      [self._body_id(name) for name in G1_COMPARISON_KEY_BODY_NAMES],
      dtype=np.int64,
    )
    joint_ids = np.asarray(
      [self._joint_id(name) for name in self.joint_names], dtype=np.int64
    )
    self.joint_qpos_addresses = self.model.jnt_qposadr[joint_ids]

    free_joint_ids = np.flatnonzero(
      self.model.jnt_type == int(mujoco.mjtJoint.mjJNT_FREE)
    )
    if len(free_joint_ids) != 1:
      raise ValueError(
        f"Expected one free joint in the G1 model, found {len(free_joint_ids)}"
      )
    self.free_qpos_address = int(self.model.jnt_qposadr[free_joint_ids[0]])
    self.neutral_qpos = np.asarray(self.model.qpos0, dtype=np.float64).copy()
    self.neutral_qpos[self.free_qpos_address : self.free_qpos_address + 7] = (
      0.0,
      0.0,
      0.0,
      1.0,
      0.0,
      0.0,
      0.0,
    )

  def _body_id(self, name: str) -> int:
    body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
      raise ValueError(f"G1 model is missing body `{name}`.")
    return body_id

  def _joint_id(self, name: str) -> int:
    joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id < 0:
      raise ValueError(f"G1 model is missing joint `{name}`.")
    return joint_id

  def validate_joint_names(self, joint_names: list[str]) -> None:
    if tuple(joint_names) != self.joint_names:
      raise ValueError("Hardware joint names differ between logs.")

  def poses_in_pelvis(
    self, joint_positions: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    num_frames = len(joint_positions)
    num_bodies = len(self.key_body_ids)
    positions = np.empty((num_frames, num_bodies, 3), dtype=np.float64)
    rotations = np.empty((num_frames, num_bodies, 3, 3), dtype=np.float64)

    for frame_index, frame_joint_pos in enumerate(joint_positions):
      self.data.qpos[:] = self.neutral_qpos
      self.data.qpos[self.joint_qpos_addresses] = frame_joint_pos
      mujoco.mj_forward(self.model, self.data)

      pelvis_pos = np.asarray(self.data.xpos[self.pelvis_body_id])
      pelvis_rot = np.asarray(self.data.xmat[self.pelvis_body_id]).reshape(3, 3)
      body_pos = np.asarray(self.data.xpos[self.key_body_ids])
      body_rot = np.asarray(self.data.xmat[self.key_body_ids]).reshape(-1, 3, 3)
      positions[frame_index] = (body_pos - pelvis_pos) @ pelvis_rot
      rotations[frame_index] = np.einsum("ij,bjk->bik", pelvis_rot.T, body_rot)

    return positions, rotations

  def tracking_errors(
    self, measured_joint_pos: np.ndarray, reference_joint_pos: np.ndarray
  ) -> tuple[np.ndarray, np.ndarray]:
    measured_pos, measured_rot = self.poses_in_pelvis(measured_joint_pos)
    reference_pos, reference_rot = self.poses_in_pelvis(reference_joint_pos)
    position_error = np.linalg.norm(measured_pos - reference_pos, axis=-1)
    rotation_delta = np.einsum("...ji,...jk->...ik", reference_rot, measured_rot)
    rotation_cosine = np.clip(
      (np.trace(rotation_delta, axis1=-2, axis2=-1) - 1.0) / 2.0,
      -1.0,
      1.0,
    )
    return position_error, np.arccos(rotation_cosine)


def _joint_groups(joint_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
  lower = np.asarray(
    [
      index
      for index, name in enumerate(joint_names)
      if any(part in name for part in ("hip", "knee", "ankle"))
    ],
    dtype=np.int64,
  )
  lower_set = set(lower.tolist())
  upper = np.asarray(
    [index for index in range(len(joint_names)) if index not in lower_set],
    dtype=np.int64,
  )
  return lower, upper


def _analyze_log(
  path: Path,
  data: np.lib.npyio.NpzFile,
  forward_kinematics: G1ForwardKinematics,
) -> dict[str, object]:
  mask = _motion_mask(data)
  if not np.any(mask):
    raise ValueError("log contains no samples in its motion phase")

  joint_names = [str(name) for name in data["meta_hardware_joint_names"]]
  measured_pos = np.asarray(data["joint_positions_hw"], dtype=np.float64)[mask]
  measured_vel = np.asarray(data["joint_velocities_hw"], dtype=np.float64)[mask]
  measured_torque = np.asarray(data["joint_torques_hw"], dtype=np.float64)[mask]
  desired_pos = np.asarray(data["desired_q_hw"], dtype=np.float64)[mask]
  reference_pos = _reference_in_hardware_order(
    data, "reference_joint_pos", joint_names
  )[mask]
  reference_vel = _reference_in_hardware_order(
    data, "reference_joint_vel", joint_names
  )[mask]
  forward_kinematics.validate_joint_names(joint_names)
  body_pos_error, body_rot_error = forward_kinematics.tracking_errors(
    measured_pos, reference_pos
  )
  lower_indices, upper_indices = _joint_groups(joint_names)

  control_dt = float(_scalar(data, "meta_control_dt", np.nan))
  reference_duration = float(
    _scalar(data, "meta_motion_length_s", len(measured_pos) * control_dt)
  )
  row: dict[str, object] = {
    "path": str(path),
    "motion": str(_scalar(data, "meta_motion_name", "unknown")),
    "policy": str(_scalar(data, "meta_onnx_name", "unknown")),
    "samples": len(measured_pos),
    "reference_duration_s": reference_duration,
    "control_dt_s": control_dt,
  }
  row.update(_prefixed("key_body_pos", _scalar_error_stats(body_pos_error)))
  row.update(_prefixed("key_body_rot", _scalar_error_stats(body_rot_error)))
  row.update(_prefixed("joint_pos", _tracking_stats(measured_pos - reference_pos)))
  row.update(_prefixed("joint_vel", _tracking_stats(measured_vel - reference_vel)))
  row.update(_prefixed("servo_pos", _tracking_stats(measured_pos - desired_pos)))
  row.update(_prefixed("torque", _torque_stats(measured_torque)))
  row.update(
    _prefixed("lower_torque", _torque_stats(measured_torque[:, lower_indices]))
  )
  row.update(
    _prefixed("upper_torque", _torque_stats(measured_torque[:, upper_indices]))
  )
  return row


def _mean_std(values: list[float]) -> tuple[float, float]:
  array = np.asarray(values, dtype=np.float64)
  std = float(np.std(array, ddof=1)) if len(array) > 1 else 0.0
  return float(np.mean(array)), std


def _motion_balanced_values(rows: list[dict[str, object]], metric: str) -> list[float]:
  grouped: dict[str, list[float]] = defaultdict(list)
  for row in rows:
    grouped[str(row["motion"])].append(float(row[metric]))
  return [float(np.mean(values)) for values in grouped.values()]


def _print_summary(
  rows: list[dict[str, object]], exclusion_counts: Counter[str], discovered: int
) -> None:
  motion_counts = Counter(str(row["motion"]) for row in rows)
  control_dts = np.asarray([float(row["control_dt_s"]) for row in rows])
  finite_dts = control_dts[np.isfinite(control_dts) & (control_dts > 0.0)]

  print("Selection")
  print(f"  Discovered NPZ logs             : {discovered}")
  print(f"  Excluded soft-ground logs       : {exclusion_counts['soft_ground']}")
  print(f"  Excluded StiffPD logs           : {exclusion_counts['stiff_pd']}")
  print(f"  Excluded incomplete logs        : {exclusion_counts['incomplete']}")
  print(f"  Included completed executions   : {len(rows)}")
  print(f"  Unique reference motions        : {len(motion_counts)}")
  print(f"  FK key bodies                   : {len(G1_COMPARISON_KEY_BODY_NAMES)}")
  print(f"  Motion-phase samples            : {sum(int(r['samples']) for r in rows)}")
  print(
    "  Accumulated reference duration  : "
    f"{sum(float(r['reference_duration_s']) for r in rows):.3f} s"
  )
  if len(finite_dts):
    print(f"  Nominal policy rate             : {1.0 / np.median(finite_dts):.3f} Hz")

  print("\nReference motions")
  for motion, count in sorted(motion_counts.items()):
    noun = "execution" if count == 1 else "executions"
    print(f"  {motion:<48} {count:>2} {noun}")

  print("\nAggregate metrics")
  print("  Each metric is computed per execution. Motion-balanced values first")
  print("  average repetitions of each motion, then summarize across motions.")
  print(
    f"  {'Metric':<30} {'Unit':<7} "
    f"{'Motion-balanced mean +/- std':>30} "
    f"{'Execution mean +/- std':>28}"
  )
  print(f"  {'-' * 30} {'-' * 7} {'-' * 30} {'-' * 28}")
  for metric, unit in METRIC_UNITS.items():
    motion_mean, motion_std = _mean_std(_motion_balanced_values(rows, metric))
    trial_mean, trial_std = _mean_std([float(row[metric]) for row in rows])
    print(
      f"  {metric:<30} {unit:<7} "
      f"{motion_mean:>12.6f} +/- {motion_std:<12.6f} "
      f"{trial_mean:>11.6f} +/- {trial_std:<10.6f}"
    )


def _write_csv(path: Path, rows: list[dict[str, object]]) -> None:
  path = path.expanduser().resolve()
  path.parent.mkdir(parents=True, exist_ok=True)
  with path.open("w", newline="", encoding="utf-8") as output:
    writer = csv.DictWriter(output, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
  print(f"\nPer-execution CSV: {path}")


def run(args: argparse.Namespace) -> None:
  log_dir = args.log_dir.expanduser().resolve()
  if not log_dir.is_dir():
    raise NotADirectoryError(f"Log directory not found: {log_dir}")

  paths = sorted(log_dir.rglob("*.npz"))
  rows: list[dict[str, object]] = []
  exclusion_counts: Counter[str] = Counter()
  forward_kinematics: G1ForwardKinematics | None = None
  for path in paths:
    with np.load(path, allow_pickle=False) as data:
      if str(_scalar(data, "meta_episode_type")) != "motion_imitation":
        exclusion_counts["other_episode_type"] += 1
        continue
      if "soft_ground" in str(path).lower():
        exclusion_counts["soft_ground"] += 1
        continue
      policy = str(_scalar(data, "meta_onnx_name", "")).lower()
      if "stiff" in policy or "stiffpd" in str(path).lower():
        exclusion_counts["stiff_pd"] += 1
        continue
      if not bool(_scalar(data, "meta_completed", False)):
        exclusion_counts["incomplete"] += 1
        continue
      try:
        joint_names = [str(name) for name in data["meta_hardware_joint_names"]]
        if forward_kinematics is None:
          forward_kinematics = G1ForwardKinematics(joint_names)
        rows.append(_analyze_log(path, data, forward_kinematics))
      except (KeyError, ValueError) as error:
        exclusion_counts["invalid"] += 1
        print(f"Skipping invalid log {path}: {error}")

  if not rows:
    raise RuntimeError("No completed logs remain after filtering.")
  _print_summary(rows, exclusion_counts, len(paths))
  if args.csv is not None:
    _write_csv(args.csv, rows)


def main() -> None:
  run(parse_args())


if __name__ == "__main__":
  main()
