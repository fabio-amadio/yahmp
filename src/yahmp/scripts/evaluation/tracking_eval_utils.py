"""Shared utilities for ONNX tracking/success evaluation scripts."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np
from mjlab.tasks.registry import load_env_cfg

from yahmp.mdp.motion.library import MotionLibrary
from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import (
  MotionClip,
  _lerp,
  _quat_conj,
  _quat_mul,
  _quat_normalize,
  _resolve_id,
)

TRACKING_METRIC_NAMES = (
  "error_anchor_pos",
  "error_anchor_rot",
  "error_anchor_lin_vel",
  "error_anchor_ang_vel",
  "error_body_pos",
  "error_body_rot",
  "error_body_lin_vel",
  "error_body_ang_vel",
  "error_joint_pos",
  "error_joint_vel",
)

DEFAULT_GROUND_FAIL_BODY_NAMES = ("torso_link",)

DEFAULT_HAND_PUSH_BODY_NAMES = (
  "left_wrist_yaw_link",
  "right_wrist_yaw_link",
)


@dataclass(frozen=True)
class HandPushMirrorConfig:
  enabled: bool = False
  body_names: tuple[str, ...] = DEFAULT_HAND_PUSH_BODY_NAMES
  duration_s: tuple[float, float] = (0.5, 2.0)
  cooldown_s: tuple[float, float] = (0.0, 0.5)
  feasible_force_fraction_range: tuple[float, float] = (0.05, 0.25)
  max_force_magnitude: float = 20.0
  force_ramp_time_fraction: float = 0.15
  dirichlet_alpha: float = 1.0
  body_point_offset: tuple[float, float, float] | None = (0.08, 0.0, 0.0)
  randomize_application_point: bool = True
  application_point_delta_range: (
    tuple[
      tuple[float, float],
      tuple[float, float],
      tuple[float, float],
    ]
    | None
  ) = (
    (-0.04, 0.06),
    (-0.01, 0.01),
    (-0.03, 0.03),
  )
  randomize_body: bool = True
  seed: int = 0


class HandPushMirror:
  """Evaluator-side mirror of YAHMP's wrist push disturbance."""

  def __init__(
    self,
    *,
    model: mujoco.MjModel,
    body_ids: np.ndarray,
    dt: float,
    cfg: HandPushMirrorConfig,
    seed: int,
  ) -> None:
    self.model = model
    self.body_ids = np.asarray(body_ids, dtype=np.int32)
    self.dt = float(dt)
    self.cfg = cfg
    self.rng = np.random.default_rng(int(seed))
    self.active = False
    self.cooldown_left = 0.0
    self.active_elapsed = 0.0
    self.active_duration = 0.0
    self.active_body_slot = -1
    self.peak_force = np.zeros(3, dtype=np.float64)
    self.peak_torque = np.zeros(3, dtype=np.float64)
    self.anchor_body_pos_w = np.full(3, np.nan, dtype=np.float64)
    self.anchor_ref_body_pos_w = np.full(3, np.nan, dtype=np.float64)
    self.disp_sum = 0.0
    self.disp_max = 0.0
    self.disp_samples = 0
    self.push_episodes = 0
    self._sample_cooldown()

  def reset(self) -> None:
    self.active = False
    self.cooldown_left = 0.0
    self.active_elapsed = 0.0
    self.active_duration = 0.0
    self.active_body_slot = -1
    self.peak_force[:] = 0.0
    self.peak_torque[:] = 0.0
    self.anchor_body_pos_w[:] = np.nan
    self.anchor_ref_body_pos_w[:] = np.nan
    self.disp_sum = 0.0
    self.disp_max = 0.0
    self.disp_samples = 0
    self.push_episodes = 0
    self._sample_cooldown()

  def clear_wrench(self, data: mujoco.MjData) -> None:
    if self.body_ids.size == 0:
      return
    data.xfrc_applied[self.body_ids, :] = 0.0

  def pre_step(
    self,
    data: mujoco.MjData,
    ref_body_pos_w_by_slot: np.ndarray | None = None,
  ) -> None:
    self.clear_wrench(data)
    if not self.cfg.enabled or self.body_ids.size == 0:
      return

    if self.active:
      self._write_active_wrench(data)
      return

    self.cooldown_left -= self.dt
    if self.cooldown_left > 0.0:
      return

    self._activate(data, ref_body_pos_w_by_slot)
    self._write_active_wrench(data)

  def post_step(
    self,
    data: mujoco.MjData,
    ref_body_pos_w_by_slot: np.ndarray | None = None,
  ) -> None:
    self.clear_wrench(data)
    if not self.cfg.enabled or not self.active or self.active_body_slot < 0:
      return

    body_id = int(self.body_ids[self.active_body_slot])
    current_pos_w = np.asarray(data.xpos[body_id], dtype=np.float64)
    robot_delta_w = current_pos_w - self.anchor_body_pos_w
    ref_delta_w = np.zeros(3, dtype=np.float64)
    if (
      ref_body_pos_w_by_slot is not None
      and self.active_body_slot < int(ref_body_pos_w_by_slot.shape[0])
      and np.all(np.isfinite(self.anchor_ref_body_pos_w))
    ):
      current_ref_body_pos_w = np.asarray(
        ref_body_pos_w_by_slot[self.active_body_slot],
        dtype=np.float64,
      )
      ref_delta_w = current_ref_body_pos_w - self.anchor_ref_body_pos_w
    disp = float(np.linalg.norm(robot_delta_w - ref_delta_w))
    self.disp_sum += disp
    self.disp_samples += 1
    self.disp_max = max(self.disp_max, disp)

    self.active_elapsed += self.dt
    if self.active_elapsed >= self.active_duration:
      self.active = False
      self.active_elapsed = 0.0
      self.active_duration = 0.0
      self.active_body_slot = -1
      self.peak_force[:] = 0.0
      self.peak_torque[:] = 0.0
      self.anchor_body_pos_w[:] = np.nan
      self.anchor_ref_body_pos_w[:] = np.nan
      self._sample_cooldown()

  def mean_displacement(self) -> float:
    if self.disp_samples <= 0:
      return math.nan
    return self.disp_sum / float(self.disp_samples)

  def max_displacement(self) -> float:
    if self.disp_samples <= 0:
      return math.nan
    return self.disp_max

  def _sample_range(self, value_range: tuple[float, float]) -> float:
    low, high = value_range
    return float(self.rng.uniform(low, high))

  def _sample_cooldown(self) -> None:
    self.cooldown_left = self._sample_range(self.cfg.cooldown_s)

  def _activate(
    self,
    data: mujoco.MjData,
    ref_body_pos_w_by_slot: np.ndarray | None,
  ) -> None:
    if self.cfg.randomize_body:
      body_slot = int(self.rng.integers(0, len(self.body_ids)))
    else:
      body_slot = 0
    body_id = int(self.body_ids[body_slot])

    offset_local = np.zeros(3, dtype=np.float64)
    if self.cfg.body_point_offset is not None:
      offset_local = np.asarray(self.cfg.body_point_offset, dtype=np.float64)
    if (
      self.cfg.randomize_application_point
      and self.cfg.application_point_delta_range is not None
    ):
      ranges = np.asarray(self.cfg.application_point_delta_range, dtype=np.float64)
      jitter = self.rng.uniform(ranges[:, 0], ranges[:, 1])
      offset_local = offset_local + jitter

    body_link_pos_w = np.asarray(data.xpos[body_id], dtype=np.float64)
    body_quat_w = np.asarray(data.xquat[body_id], dtype=np.float64)
    body_com_pos_w = np.asarray(data.xipos[body_id], dtype=np.float64)
    offset_w = _quat_rotate(body_quat_w, offset_local)
    point_w = body_link_pos_w + offset_w
    moment_arm_w = point_w - body_com_pos_w

    alpha = max(float(self.cfg.dirichlet_alpha), 1.0e-6)
    weights = self.rng.dirichlet(np.full(3, alpha, dtype=np.float64))
    signs = self.rng.choice(np.asarray([-1.0, 1.0], dtype=np.float64), size=3)
    direction = signs * weights
    direction_norm = np.linalg.norm(direction)
    if direction_norm < 1.0e-12:
      direction = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    else:
      direction = direction / direction_norm

    force_fraction = self._sample_range(self.cfg.feasible_force_fraction_range)
    force_mag = float(self.cfg.max_force_magnitude) * force_fraction
    force = direction * force_mag
    torque = np.cross(moment_arm_w, force)

    self.active = True
    self.active_elapsed = 0.0
    self.active_duration = self._sample_range(self.cfg.duration_s)
    self.active_body_slot = body_slot
    self.peak_force = force
    self.peak_torque = torque
    self.anchor_body_pos_w = body_link_pos_w.copy()
    if ref_body_pos_w_by_slot is not None and body_slot < int(
      ref_body_pos_w_by_slot.shape[0]
    ):
      self.anchor_ref_body_pos_w = np.asarray(
        ref_body_pos_w_by_slot[body_slot],
        dtype=np.float64,
      ).copy()
    else:
      self.anchor_ref_body_pos_w[:] = np.nan
    self.push_episodes += 1

  def _write_active_wrench(self, data: mujoco.MjData) -> None:
    if self.active_body_slot < 0:
      return
    scale = self._compute_force_ramp_scale()
    body_id = int(self.body_ids[self.active_body_slot])
    data.xfrc_applied[body_id, :3] = self.peak_force * scale
    data.xfrc_applied[body_id, 3:] = self.peak_torque * scale

  def _compute_force_ramp_scale(self) -> float:
    ramp = float(self.cfg.force_ramp_time_fraction)
    if ramp <= 0.0 or self.active_duration <= 1.0e-12:
      return 1.0
    phase = min(
      max((self.active_elapsed + 0.5 * self.dt) / self.active_duration, 0.0),
      1.0,
    )
    ramp_up = min(phase / ramp, 1.0)
    ramp_down = min((1.0 - phase) / ramp, 1.0)
    return float(max(min(ramp_up, ramp_down), 0.0))


def sample_clip_body_positions_w(
  clip: MotionClip,
  time_s: float,
  body_names: tuple[str, ...],
) -> np.ndarray:
  """Return reference body positions in world frame for the requested bodies."""
  positions = []
  safe_time_s = safe_motion_time(clip, time_s)
  for body_name in body_names:
    body_index = clip.body_index(body_name)
    pos_w, _ = clip.sample_body_pose(safe_time_s, body_index)
    positions.append(np.asarray(pos_w, dtype=np.float64))
  if not positions:
    return np.zeros((0, 3), dtype=np.float64)
  return np.asarray(positions, dtype=np.float64)


def resolve_motion_source(task_id: str, override: str | None) -> str:
  if override is not None:
    return str(Path(override).expanduser().resolve())

  env_cfg = load_env_cfg(task_id, play=True)
  motion_cfg = env_cfg.commands.get("motion")
  if motion_cfg is None:
    raise ValueError(f"Task does not define a `motion` command: {task_id}")
  motion_source = getattr(motion_cfg, "motion_file", None)
  if not isinstance(motion_source, str) or motion_source == "":
    raise ValueError(
      f"Task does not define a non-empty motion source in its `motion` command: {task_id}"
    )
  return str(Path(motion_source).expanduser().resolve())


def resolve_motion_files(motion_source: str) -> list[Path]:
  motion_files, _ = MotionLibrary._resolve_motion_entries(motion_source)
  return [path.expanduser().resolve() for path in motion_files]


def safe_motion_time(clip: MotionClip, time_s: float) -> float:
  if clip.length_s <= 0.0:
    return 0.0
  return min(max(float(time_s), 0.0), max(clip.length_s - 1.0e-9, 0.0))


def body_velocity_w(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
  velocity = np.zeros(6, dtype=np.float64)
  mujoco.mj_objectVelocity(
    model,
    data,
    mujoco.mjtObj.mjOBJ_BODY,
    body_id,
    velocity,
    0,
  )
  return velocity[3:6].copy(), velocity[0:3].copy()


def resolve_body_ids(
  model: mujoco.MjModel,
  body_names: tuple[str, ...],
) -> np.ndarray:
  return np.asarray(
    [_resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, name) for name in body_names],
    dtype=np.int32,
  )


def ground_contact_active(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  fail_body_ids: np.ndarray,
) -> bool:
  if fail_body_ids.size == 0:
    return False
  fail_set = set(int(idx) for idx in fail_body_ids)
  for contact_idx in range(int(data.ncon)):
    contact = data.contact[contact_idx]
    body_1 = int(model.geom_bodyid[int(contact.geom1)])
    body_2 = int(model.geom_bodyid[int(contact.geom2)])
    if (body_1 in fail_set and body_2 == 0) or (body_2 in fail_set and body_1 == 0):
      return True
  return False


def compute_tracking_metrics(
  *,
  model: mujoco.MjModel,
  data: mujoco.MjData,
  clip: MotionClip,
  time_s: float,
  joint_qpos_adr: np.ndarray,
  joint_qvel_adr: np.ndarray,
  root_body_name: str,
  key_body_names: tuple[str, ...],
) -> dict[str, float]:
  """Compute the canonical YAHMP motion-command tracking metrics."""
  time_s = safe_motion_time(clip, time_s)
  frame = clip.sample(time_s)
  root_body_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
  robot_root_pos = np.asarray(data.xpos[root_body_id], dtype=np.float64)
  robot_root_quat = np.asarray(data.xquat[root_body_id], dtype=np.float64)
  robot_root_lin_vel, robot_root_ang_vel = body_velocity_w(model, data, root_body_id)
  delta_pos_w = robot_root_pos.copy()
  delta_pos_w[2] = frame.root_pos_w[2]
  delta_ori_w = _yaw_quat(_quat_mul(robot_root_quat, _quat_conj(frame.root_quat_w)))

  body_pos_errors = []
  body_rot_errors = []
  body_lin_vel_errors = []
  body_ang_vel_errors = []
  for body_name in key_body_names:
    body_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    body_index = clip.body_index(body_name)
    ref_pos, ref_quat = clip.sample_body_pose(time_s, body_index)
    ref_lin_vel, ref_ang_vel = _sample_body_velocity(clip, time_s, body_index)
    robot_lin_vel, robot_ang_vel = body_velocity_w(model, data, body_id)
    ref_pos_relative_w = delta_pos_w + _quat_rotate(
      delta_ori_w, ref_pos - frame.root_pos_w
    )
    ref_quat_relative_w = _quat_mul(delta_ori_w, ref_quat)
    body_pos_errors.append(
      float(
        np.linalg.norm(
          np.asarray(data.xpos[body_id], dtype=np.float64) - ref_pos_relative_w
        )
      )
    )
    body_rot_errors.append(
      _quat_angle(
        ref_quat_relative_w, np.asarray(data.xquat[body_id], dtype=np.float64)
      )
    )
    body_lin_vel_errors.append(float(np.linalg.norm(robot_lin_vel - ref_lin_vel)))
    body_ang_vel_errors.append(float(np.linalg.norm(robot_ang_vel - ref_ang_vel)))

  joint_pos = np.asarray(data.qpos[joint_qpos_adr], dtype=np.float64)
  joint_vel = np.asarray(data.qvel[joint_qvel_adr], dtype=np.float64)
  joint_pos_err = joint_pos - frame.joint_pos
  joint_vel_err = joint_vel - frame.joint_vel

  return {
    "error_anchor_pos": float(np.linalg.norm(robot_root_pos - frame.root_pos_w)),
    "error_anchor_rot": _quat_angle(frame.root_quat_w, robot_root_quat),
    "error_anchor_lin_vel": float(
      np.linalg.norm(robot_root_lin_vel - frame.root_lin_vel_w)
    ),
    "error_anchor_ang_vel": float(
      np.linalg.norm(robot_root_ang_vel - frame.root_ang_vel_w)
    ),
    "error_body_pos": _nanmean(body_pos_errors),
    "error_body_rot": _nanmean(body_rot_errors),
    "error_body_lin_vel": _nanmean(body_lin_vel_errors),
    "error_body_ang_vel": _nanmean(body_ang_vel_errors),
    "error_joint_pos": float(np.linalg.norm(joint_pos_err)),
    "error_joint_vel": float(np.linalg.norm(joint_vel_err)),
  }


def _nanmean(values: list[float]) -> float:
  finite_values = [value for value in values if math.isfinite(value)]
  if len(finite_values) == 0:
    return math.nan
  return sum(finite_values) / float(len(finite_values))


def _quat_angle(q_ref: np.ndarray, q_robot: np.ndarray) -> float:
  q_ref = _quat_normalize(np.asarray(q_ref, dtype=np.float64))
  q_robot = _quat_normalize(np.asarray(q_robot, dtype=np.float64))
  q_err = _quat_mul(_quat_conj(q_ref), q_robot)
  return float(2.0 * np.arctan2(np.linalg.norm(q_err[1:]), abs(q_err[0])))


def _quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  q = _quat_normalize(np.asarray(q, dtype=np.float64))
  v = np.asarray(v, dtype=np.float64)
  qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
  return _quat_mul(_quat_mul(q, qv), _quat_conj(q))[1:]


def _yaw_quat(q: np.ndarray) -> np.ndarray:
  q = _quat_normalize(np.asarray(q, dtype=np.float64))
  w, x, y, z = q
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  half_yaw = 0.5 * yaw
  return np.array([np.cos(half_yaw), 0.0, 0.0, np.sin(half_yaw)], dtype=np.float64)


def _sample_body_velocity(
  clip: MotionClip,
  time_s: float,
  body_index: int,
) -> tuple[np.ndarray, np.ndarray]:
  idx0, idx1, blend = clip._sample_indices(time_s)
  return (
    _lerp(
      clip.body_lin_vel_w[idx0, body_index],
      clip.body_lin_vel_w[idx1, body_index],
      blend,
    ),
    _lerp(
      clip.body_ang_vel_w[idx0, body_index],
      clip.body_ang_vel_w[idx1, body_index],
      blend,
    ),
  )


def upper_lower_body_masks(
  joint_names: tuple[str, ...],
) -> tuple[np.ndarray, np.ndarray]:
  lower_keywords = ("hip_", "knee_", "ankle_")
  lower_mask = np.asarray(
    [any(keyword in name for keyword in lower_keywords) for name in joint_names],
    dtype=bool,
  )
  upper_mask = ~lower_mask
  return upper_mask, lower_mask


def init_actuation_stats(
  num_envs: int, joint_names: tuple[str, ...]
) -> dict[str, np.ndarray | tuple[str, ...]]:
  num_joints = len(joint_names)
  return {
    "joint_names": joint_names,
    "torque_abs_sum": np.zeros((num_envs, num_joints), dtype=np.float64),
    "torque_abs_max": np.zeros((num_envs, num_joints), dtype=np.float64),
    "sample_count": np.zeros(num_envs, dtype=np.int64),
  }


def update_actuation_stats(
  stats: dict[str, np.ndarray | tuple[str, ...]],
  env_idx: int,
  torque: np.ndarray,
) -> None:
  torque = np.asarray(torque, dtype=np.float64)
  torque_abs = np.abs(torque)
  stats["torque_abs_sum"][env_idx] += torque_abs
  stats["torque_abs_max"][env_idx] = np.maximum(
    stats["torque_abs_max"][env_idx], torque_abs
  )
  stats["sample_count"][env_idx] += 1


def actuation_stats_to_row(
  stats: dict[str, np.ndarray | tuple[str, ...]],
  env_idx: int,
) -> dict[str, float]:
  joint_names = tuple(stats["joint_names"])
  sample_count = int(stats["sample_count"][env_idx])
  upper_mask, lower_mask = upper_lower_body_masks(joint_names)

  def _mean_or_nan(values: np.ndarray) -> float:
    if sample_count <= 0:
      return math.nan
    return float(values[env_idx].mean() / float(sample_count))

  def _group_mean_or_nan(values: np.ndarray, mask: np.ndarray) -> float:
    if sample_count <= 0 or not np.any(mask):
      return math.nan
    return float(values[env_idx, mask].mean() / float(sample_count))

  def _group_max_or_nan(values: np.ndarray, mask: np.ndarray) -> float:
    if sample_count <= 0 or not np.any(mask):
      return math.nan
    return float(values[env_idx, mask].max())

  row: dict[str, float] = {
    "avg_abs_torque_all": _mean_or_nan(stats["torque_abs_sum"]),
    "max_abs_torque_all": float(stats["torque_abs_max"][env_idx].max())
    if sample_count > 0
    else math.nan,
    "avg_abs_torque_upper": _group_mean_or_nan(stats["torque_abs_sum"], upper_mask),
    "max_abs_torque_upper": _group_max_or_nan(stats["torque_abs_max"], upper_mask),
    "avg_abs_torque_lower": _group_mean_or_nan(stats["torque_abs_sum"], lower_mask),
    "max_abs_torque_lower": _group_max_or_nan(stats["torque_abs_max"], lower_mask),
  }
  return row
