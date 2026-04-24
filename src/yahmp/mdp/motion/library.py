"""Motion-data loading and library query utilities for YAHMP."""

from __future__ import annotations

import json
import os
import pickle
import sys
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .indexing import build_name_to_index

try:
  from tqdm import tqdm
except ModuleNotFoundError:
  tqdm = None


class MotionLoader:
  """Load a single motion file into torch tensors."""

  def __init__(
    self,
    motion_file: str,
    device: str = "cpu",
    required_body_names: tuple[str, ...] | None = None,
  ) -> None:
    payload = load_motion_file(motion_file, required_body_names=required_body_names)

    self.joint_pos = torch.tensor(payload.joint_pos, dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(payload.joint_vel, dtype=torch.float32, device=device)
    self.body_pos_w = torch.tensor(
      payload.body_pos_w, dtype=torch.float32, device=device
    )
    self.body_quat_w = torch.tensor(
      payload.body_quat_w, dtype=torch.float32, device=device
    )
    self.body_lin_vel_w = torch.tensor(
      payload.body_lin_vel_w, dtype=torch.float32, device=device
    )
    self.body_ang_vel_w = torch.tensor(
      payload.body_ang_vel_w, dtype=torch.float32, device=device
    )
    self.body_names = payload.body_names

    self.time_step_total = self.joint_pos.shape[0]

  @staticmethod
  def _decode_name(value: object) -> str:
    if isinstance(value, bytes):
      return value.decode("utf-8")
    return str(value)

  @classmethod
  def _extract_body_names(cls, data) -> tuple[str, ...] | None:
    if "body_names" in data:
      flat_values = np.asarray(data["body_names"]).reshape(-1)
      return tuple(cls._decode_name(v) for v in flat_values.tolist())
    return None


@dataclass
class MotionFrameBatch:
  """Batch of queried motion-reference tensors in [env, ...] form."""

  joint_pos: torch.Tensor
  joint_vel: torch.Tensor
  body_pos_w: torch.Tensor
  body_quat_w: torch.Tensor
  body_lin_vel_w: torch.Tensor
  body_ang_vel_w: torch.Tensor
  anchor_pos_w: torch.Tensor
  anchor_quat_w: torch.Tensor
  anchor_lin_vel_w: torch.Tensor
  anchor_ang_vel_w: torch.Tensor


@dataclass(frozen=True)
class LoadedMotionData:
  """Numpy-backed motion arrays in the canonical YAHMP format."""

  fps: float
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  body_pos_w: np.ndarray
  body_quat_w: np.ndarray
  body_lin_vel_w: np.ndarray
  body_ang_vel_w: np.ndarray
  body_names: tuple[str, ...]


@dataclass(frozen=True)
class RawMotionData:
  """Partially specified motion arrays before FK/velocity completion."""

  fps: float
  joint_pos: np.ndarray
  joint_vel: np.ndarray | None = None
  root_pos: np.ndarray | None = None
  root_quat_w: np.ndarray | None = None
  body_pos_w: np.ndarray | None = None
  body_quat_w: np.ndarray | None = None
  body_lin_vel_w: np.ndarray | None = None
  body_ang_vel_w: np.ndarray | None = None
  body_names: tuple[str, ...] | None = None


def load_motion_file(
  motion_file: str | Path,
  *,
  required_body_names: tuple[str, ...] | None = None,
) -> LoadedMotionData:
  path = Path(motion_file).expanduser()
  if path.suffix == ".npz":
    payload = _load_npz_motion(path)
  elif path.suffix in (".pkl", ".pickle"):
    payload = _load_twist2_pkl_motion(path)
  else:
    raise ValueError(f"Unsupported motion file extension: {path}")

  selected_indices: np.ndarray | None = None
  selected_body_names = payload.body_names
  if required_body_names is not None:
    selected_indices = _resolve_required_body_indices(
      file_body_names=payload.body_names,
      required_body_names=required_body_names,
      source=path,
    )
    selected_body_names = tuple(required_body_names)

  if selected_indices is None:
    return payload

  return LoadedMotionData(
    fps=payload.fps,
    joint_pos=payload.joint_pos,
    joint_vel=payload.joint_vel,
    body_pos_w=payload.body_pos_w[:, selected_indices, :],
    body_quat_w=payload.body_quat_w[:, selected_indices, :],
    body_lin_vel_w=payload.body_lin_vel_w[:, selected_indices, :],
    body_ang_vel_w=payload.body_ang_vel_w[:, selected_indices, :],
    body_names=selected_body_names,
  )


def _env_int(name: str, default: int) -> int:
  value = os.environ.get(name)
  if value is None:
    return default
  try:
    return int(value)
  except ValueError:
    return default


def _should_show_progress(show_progress: bool | None) -> bool:
  if show_progress is not None:
    return show_progress
  return (
    _env_int("RANK", 0) == 0 and _env_int("LOCAL_RANK", 0) == 0 and sys.stderr.isatty()
  )


def _load_yaml_config(yaml_path: Path) -> dict[str, object]:
  import yaml

  with open(yaml_path, "r", encoding="utf-8") as f:
    data = yaml.safe_load(f)
  if not isinstance(data, dict):
    raise ValueError(f"YAML config must be a mapping at top level: {yaml_path}")
  return data


def _load_sidecar_motion_meta(path: Path) -> dict[str, Any] | None:
  meta_path = path.with_name("meta.json")
  if not meta_path.exists():
    return None
  meta = json.loads(meta_path.read_text(encoding="utf-8"))
  if not isinstance(meta, dict):
    raise ValueError(f"Expected object/dict in sidecar motion metadata: {meta_path}")
  return meta


def _extract_npz_fps(data: Any, sidecar_meta: dict[str, Any] | None) -> float:
  if "fps" not in data:
    if sidecar_meta is not None and "fps" in sidecar_meta:
      return float(sidecar_meta["fps"])
    return 30.0
  fps_value = np.asarray(data["fps"]).reshape(-1)
  if fps_value.size == 0:
    return 30.0
  return float(fps_value[0])


def _extract_npz_body_names(
  data: Any,
  sidecar_meta: dict[str, Any] | None,
) -> tuple[str, ...] | None:
  names = MotionLoader._extract_body_names(data)
  if names is not None:
    return names
  if sidecar_meta is not None and "body_names" in sidecar_meta:
    return tuple(str(value) for value in sidecar_meta["body_names"])
  return None


def _quat_convention_from_env(*, env_name: str | None, default: str) -> str:
  if env_name is not None:
    env_value = os.environ.get(env_name)
    if env_value:
      return env_value.lower()
  return default


def _normalize_fps(fps_value: object, default: float = 30.0) -> float:
  fps = float(np.asarray(fps_value).reshape(-1)[0])
  if fps <= 1.0e-6:
    return default
  return fps


def _extract_optional_array(
  data: Any,
  *keys: str,
  dtype: np.dtype | type,
) -> np.ndarray | None:
  for key in keys:
    if key in data:
      return np.asarray(data[key], dtype=dtype)
  return None


def _validate_joint_root_shapes(
  *,
  source: Path,
  joint_pos: np.ndarray,
  joint_vel: np.ndarray | None,
  root_pos: np.ndarray | None,
  root_quat_w: np.ndarray | None,
) -> None:
  if joint_pos.ndim != 2:
    raise ValueError(f"Invalid joint_pos shape in {source}: {joint_pos.shape}")
  num_frames = joint_pos.shape[0]
  if joint_vel is not None and joint_vel.shape != joint_pos.shape:
    raise ValueError(
      f"Invalid joint_vel shape in {source}: {joint_vel.shape}, expected {joint_pos.shape}"
    )
  if root_pos is not None and (root_pos.ndim != 2 or root_pos.shape != (num_frames, 3)):
    raise ValueError(f"Invalid root_pos shape in {source}: {root_pos.shape}")
  if root_quat_w is not None and (
    root_quat_w.ndim != 2 or root_quat_w.shape != (num_frames, 4)
  ):
    raise ValueError(f"Invalid root_quat_w shape in {source}: {root_quat_w.shape}")


def _validate_body_shapes(
  *,
  source: Path,
  num_frames: int,
  body_pos_w: np.ndarray | None,
  body_quat_w: np.ndarray | None,
  body_lin_vel_w: np.ndarray | None,
  body_ang_vel_w: np.ndarray | None,
  body_names: tuple[str, ...] | None,
) -> None:
  if body_pos_w is None and body_quat_w is None:
    return
  if body_pos_w is None or body_quat_w is None:
    raise ValueError(
      f"{source} must provide both `body_pos_w` and `body_quat_w`, or neither."
    )
  if (
    body_pos_w.ndim != 3
    or body_pos_w.shape[0] != num_frames
    or body_pos_w.shape[-1] != 3
  ):
    raise ValueError(f"Invalid body_pos_w shape in {source}: {body_pos_w.shape}")
  if (
    body_quat_w.ndim != 3
    or body_quat_w.shape[:2] != body_pos_w.shape[:2]
    or body_quat_w.shape[-1] != 4
  ):
    raise ValueError(f"Invalid body_quat_w shape in {source}: {body_quat_w.shape}")
  if body_lin_vel_w is not None and body_lin_vel_w.shape != body_pos_w.shape:
    raise ValueError(
      f"Invalid body_lin_vel_w shape in {source}: {body_lin_vel_w.shape}, expected {body_pos_w.shape}"
    )
  if body_ang_vel_w is not None and (
    body_ang_vel_w.shape[:2] != body_pos_w.shape[:2] or body_ang_vel_w.shape[-1] != 3
  ):
    raise ValueError(
      f"Invalid body_ang_vel_w shape in {source}: {body_ang_vel_w.shape}"
    )
  if body_names is not None and len(body_names) != body_pos_w.shape[1]:
    raise ValueError(
      f"Invalid body_names length in {source}: {len(body_names)} "
      f"for {body_pos_w.shape[1]} bodies."
    )


def _fk_body_kinematics(
  *,
  source: Path,
  root_pos: np.ndarray,
  root_quat_w: np.ndarray,
  joint_pos: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
  import mujoco

  model, data, free_qadr, joint_q_indices, body_ids, body_names = (
    _cached_g1_fk_context()
  )
  if len(joint_q_indices) != joint_pos.shape[1]:
    raise ValueError(
      "DoF mismatch between G1 FK model and motion file: "
      f"model={len(joint_q_indices)} file={joint_pos.shape[1]} in {source}"
    )

  num_frames = joint_pos.shape[0]
  num_bodies = len(body_ids)
  body_pos_w = np.zeros((num_frames, num_bodies, 3), dtype=np.float32)
  body_quat_w = np.zeros((num_frames, num_bodies, 4), dtype=np.float32)
  qpos = np.zeros(model.nq, dtype=np.float64)
  joint_q_indices_arr = np.asarray(joint_q_indices, dtype=np.int64)

  for frame_id in range(num_frames):
    qpos[:] = 0.0
    qpos[free_qadr : free_qadr + 3] = root_pos[frame_id]
    qpos[free_qadr + 3 : free_qadr + 7] = root_quat_w[frame_id]
    qpos[joint_q_indices_arr] = joint_pos[frame_id]
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    for out_idx, body_id in enumerate(body_ids):
      body_pos_w[frame_id, out_idx] = data.xpos[body_id]
      body_quat_w[frame_id, out_idx] = data.xquat[body_id]

  return body_pos_w, _quat_normalize_np(body_quat_w).astype(np.float32), body_names


def _finalize_motion_data(raw: RawMotionData, *, source: Path) -> LoadedMotionData:
  fps = _normalize_fps(raw.fps)
  dt = 1.0 / fps

  joint_pos = np.asarray(raw.joint_pos, dtype=np.float32)
  joint_vel = (
    None if raw.joint_vel is None else np.asarray(raw.joint_vel, dtype=np.float32)
  )
  root_pos = (
    None if raw.root_pos is None else np.asarray(raw.root_pos, dtype=np.float64)
  )
  root_quat_w = (
    None
    if raw.root_quat_w is None
    else _quat_normalize_np(np.asarray(raw.root_quat_w, dtype=np.float64))
  )
  body_pos_w = (
    None if raw.body_pos_w is None else np.asarray(raw.body_pos_w, dtype=np.float32)
  )
  body_quat_w = (
    None
    if raw.body_quat_w is None
    else _quat_normalize_np(np.asarray(raw.body_quat_w, dtype=np.float32))
  )
  body_lin_vel_w = (
    None
    if raw.body_lin_vel_w is None
    else np.asarray(raw.body_lin_vel_w, dtype=np.float32)
  )
  body_ang_vel_w = (
    None
    if raw.body_ang_vel_w is None
    else np.asarray(raw.body_ang_vel_w, dtype=np.float32)
  )
  body_names = raw.body_names

  _validate_joint_root_shapes(
    source=source,
    joint_pos=joint_pos,
    joint_vel=joint_vel,
    root_pos=root_pos,
    root_quat_w=root_quat_w,
  )
  _validate_body_shapes(
    source=source,
    num_frames=joint_pos.shape[0],
    body_pos_w=body_pos_w,
    body_quat_w=body_quat_w,
    body_lin_vel_w=body_lin_vel_w,
    body_ang_vel_w=body_ang_vel_w,
    body_names=body_names,
  )

  if body_pos_w is None or body_quat_w is None:
    if root_pos is None or root_quat_w is None:
      raise ValueError(
        f"{source} is missing body/world kinematics and does not provide "
        "`root_pos` + `root_quat_w` to reconstruct them with FK."
      )
    body_pos_w, body_quat_w, body_names = _fk_body_kinematics(
      source=source,
      root_pos=root_pos,
      root_quat_w=root_quat_w,
      joint_pos=joint_pos,
    )

  assert body_names is not None
  if joint_vel is None:
    joint_vel = _finite_difference_np(joint_pos, dt)
  if body_lin_vel_w is None:
    body_lin_vel_w = _finite_difference_np(body_pos_w, dt)
  if body_ang_vel_w is None:
    body_ang_vel_w = _quat_angular_velocity_np(body_quat_w.astype(np.float64), dt)

  return LoadedMotionData(
    fps=fps,
    joint_pos=joint_pos,
    joint_vel=joint_vel.astype(np.float32),
    body_pos_w=body_pos_w.astype(np.float32),
    body_quat_w=body_quat_w.astype(np.float32),
    body_lin_vel_w=body_lin_vel_w.astype(np.float32),
    body_ang_vel_w=body_ang_vel_w.astype(np.float32),
    body_names=tuple(body_names),
  )


def _load_npz_motion(path: Path) -> LoadedMotionData:
  sidecar_meta = _load_sidecar_motion_meta(path)
  with np.load(path, allow_pickle=False) as data:
    joint_pos = _extract_optional_array(data, "joint_pos", dtype=np.float32)
    if joint_pos is None:
      raise ValueError(f"Motion npz must include `joint_pos`: {path}")
    root_quat_w = _extract_optional_array(data, "root_quat_w", dtype=np.float64)
    if root_quat_w is not None:
      root_quat_w = _to_wxyz(
        root_quat_w,
        _quat_convention_from_env(
          env_name="YAHMP_MOTION_NPZ_QUAT_CONVENTION",
          default="wxyz",
        ),
      )
    raw = RawMotionData(
      fps=_extract_npz_fps(data, sidecar_meta),
      joint_pos=joint_pos,
      joint_vel=_extract_optional_array(data, "joint_vel", dtype=np.float32),
      root_pos=_extract_optional_array(data, "root_pos", dtype=np.float64),
      root_quat_w=root_quat_w,
      body_pos_w=_extract_optional_array(data, "body_pos_w", dtype=np.float32),
      body_quat_w=_extract_optional_array(data, "body_quat_w", dtype=np.float32),
      body_lin_vel_w=_extract_optional_array(data, "body_lin_vel_w", dtype=np.float32),
      body_ang_vel_w=_extract_optional_array(data, "body_ang_vel_w", dtype=np.float32),
      body_names=_extract_npz_body_names(data, sidecar_meta),
    )
  return _finalize_motion_data(raw, source=path)


def _to_wxyz(quat: np.ndarray, convention: str) -> np.ndarray:
  convention = convention.lower()
  if convention == "wxyz":
    return quat
  if convention == "xyzw":
    return np.roll(quat, shift=1, axis=-1)
  raise ValueError(f"Unsupported quaternion convention: {convention}")


def _quat_normalize_np(q: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(q, axis=-1, keepdims=True)
  return q / np.clip(norm, 1.0e-12, None)


def _quat_conj_np(q: np.ndarray) -> np.ndarray:
  out = q.copy()
  out[..., 1:] *= -1.0
  return out


def _quat_mul_np(a: np.ndarray, b: np.ndarray) -> np.ndarray:
  aw, ax, ay, az = np.moveaxis(a, -1, 0)
  bw, bx, by, bz = np.moveaxis(b, -1, 0)
  return np.stack(
    (
      aw * bw - ax * bx - ay * by - az * bz,
      aw * bx + ax * bw + ay * bz - az * by,
      aw * by - ax * bz + ay * bw + az * bx,
      aw * bz + ax * by - ay * bx + az * bw,
    ),
    axis=-1,
  )


def _quat_to_rotvec_np(q: np.ndarray) -> np.ndarray:
  q = _quat_normalize_np(q)
  w = np.clip(q[..., 0], -1.0, 1.0)
  v = q[..., 1:]
  v_norm = np.linalg.norm(v, axis=-1, keepdims=True)
  small = v_norm < 1.0e-8

  angle = 2.0 * np.arctan2(v_norm[..., 0], w)
  axis = np.zeros_like(v)
  np.divide(v, v_norm, out=axis, where=~small)
  rotvec = axis * angle[..., None]
  return np.where(small, 2.0 * v, rotvec)


def _finite_difference_np(x: np.ndarray, dt: float) -> np.ndarray:
  if x.shape[0] < 2:
    return np.zeros_like(x, dtype=np.float32)
  return np.gradient(x, dt, axis=0).astype(np.float32)


def _quat_angular_velocity_np(quat_wxyz: np.ndarray, dt: float) -> np.ndarray:
  num_frames = quat_wxyz.shape[0]
  if num_frames < 2:
    return np.zeros((*quat_wxyz.shape[:-1], 3), dtype=np.float32)

  q_prev = quat_wxyz[:-2]
  q_next = quat_wxyz[2:].copy()
  flip = np.sum(q_next * q_prev, axis=-1, keepdims=True) < 0.0
  q_next = np.where(flip, -q_next, q_next)
  q_rel = _quat_mul_np(q_next, _quat_conj_np(q_prev))
  omega = _quat_to_rotvec_np(q_rel) / (2.0 * dt)
  omega = np.concatenate([omega[:1], omega, omega[-1:]], axis=0)
  return omega.astype(np.float32)


def _joint_qpos_indices(model: Any) -> tuple[int, list[int]]:
  import mujoco

  free_joint_qadr: int | None = None
  joint_q_indices: list[int] = []
  for jid in range(model.njnt):
    jtype = model.jnt_type[jid]
    qadr = int(model.jnt_qposadr[jid])
    if jtype == mujoco.mjtJoint.mjJNT_FREE:
      if free_joint_qadr is None:
        free_joint_qadr = qadr
      continue
    if jtype in (mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE):
      joint_q_indices.append(qadr)
    elif jtype == mujoco.mjtJoint.mjJNT_BALL:
      joint_q_indices.extend(range(qadr, qadr + 4))
    else:
      raise ValueError(f"Unsupported joint type in FK model: {jtype}")
  if free_joint_qadr is None:
    raise ValueError("FK model has no free joint.")
  return free_joint_qadr, sorted(joint_q_indices)


def _robot_body_ids_and_names(model: Any) -> tuple[list[int], tuple[str, ...]]:
  import mujoco

  body_ids: list[int] = []
  body_names: list[str] = []
  for bid in range(model.nbody):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    if name is None or name == "world":
      continue
    body_ids.append(bid)
    body_names.append(name)
  if len(body_ids) == 0:
    raise ValueError("No named non-world bodies found in FK model.")
  return body_ids, tuple(body_names)


@lru_cache(maxsize=1)
def _cached_g1_fk_context() -> tuple[
  Any, Any, int, tuple[int, ...], tuple[int, ...], tuple[str, ...]
]:
  import mujoco
  from mjlab.asset_zoo.robots.unitree_g1.g1_constants import G1_XML

  model = mujoco.MjModel.from_xml_path(str(G1_XML))
  data = mujoco.MjData(model)
  free_qadr, joint_q_indices = _joint_qpos_indices(model)
  body_ids, body_names = _robot_body_ids_and_names(model)
  return model, data, free_qadr, tuple(joint_q_indices), tuple(body_ids), body_names


def _load_pickle_dict(path: Path) -> dict[str, object]:
  with path.open("rb") as f:
    data = pickle.load(f)
  if not isinstance(data, dict):
    raise ValueError(f"Invalid PKL object type in {path}: {type(data)}")
  return data


def _pkl_quat_convention(motion: dict[str, object]) -> str:
  return _quat_convention_from_env(
    env_name="YAHMP_MOTION_PKL_QUAT_CONVENTION",
    default="xyzw",
  )


def _load_twist2_pkl_motion(path: Path) -> LoadedMotionData:
  motion = _load_pickle_dict(path)
  required_keys = {"root_pos", "root_rot", "dof_pos"}
  missing = required_keys.difference(set(motion.keys()))
  if missing:
    raise ValueError(f"Invalid motion pkl. Missing keys: {sorted(missing)} in {path}")

  root_quat = np.asarray(motion["root_rot"], dtype=np.float64)
  raw = RawMotionData(
    fps=motion.get("fps", 30.0),
    joint_pos=np.asarray(motion["dof_pos"], dtype=np.float32),
    joint_vel=_extract_optional_array(motion, "joint_vel", dtype=np.float32),
    root_pos=np.asarray(motion["root_pos"], dtype=np.float64),
    root_quat_w=_to_wxyz(root_quat, _pkl_quat_convention(motion)),
    body_pos_w=_extract_optional_array(motion, "body_pos_w", dtype=np.float32),
    body_quat_w=_extract_optional_array(motion, "body_quat_w", dtype=np.float32),
    body_lin_vel_w=_extract_optional_array(motion, "body_lin_vel_w", dtype=np.float32),
    body_ang_vel_w=_extract_optional_array(motion, "body_ang_vel_w", dtype=np.float32),
    body_names=MotionLoader._extract_body_names(motion),
  )
  return _finalize_motion_data(raw, source=path)


def _resolve_required_body_indices(
  file_body_names: tuple[str, ...],
  required_body_names: tuple[str, ...],
  source: str | Path,
) -> np.ndarray:
  if len(required_body_names) == 0:
    raise ValueError("`required_body_names` must be non-empty when provided.")
  unique_required = tuple(dict.fromkeys(required_body_names))
  if len(unique_required) != len(required_body_names):
    raise ValueError(
      f"`required_body_names` contains duplicates for `{source}`: {required_body_names}"
    )

  name_to_index = build_name_to_index(file_body_names, str(source))
  missing = [name for name in required_body_names if name not in name_to_index]
  if missing:
    raise ValueError(
      f"Missing required motion bodies in `{source}`: {missing}. "
      f"Available bodies: {file_body_names}"
    )
  return np.asarray(
    [name_to_index[name] for name in required_body_names], dtype=np.int64
  )


def _quat_slerp_batch(
  q0: torch.Tensor, q1: torch.Tensor, blend: torch.Tensor
) -> torch.Tensor:
  """Vectorized quaternion slerp for tensors with matching shape [..., 4]."""
  q0 = q0 / torch.clamp(torch.norm(q0, dim=-1, keepdim=True), min=1e-8)
  q1 = q1 / torch.clamp(torch.norm(q1, dim=-1, keepdim=True), min=1e-8)
  blend = blend.unsqueeze(-1)

  dot = torch.sum(q0 * q1, dim=-1, keepdim=True)
  q1 = torch.where(dot < 0.0, -q1, q1)
  dot = torch.abs(dot).clamp(max=1.0)

  close = dot > 0.9995
  theta_0 = torch.acos(dot)
  sin_theta_0 = torch.sin(theta_0)
  theta = theta_0 * blend
  sin_theta = torch.sin(theta)

  s0 = torch.sin(theta_0 - theta) / torch.clamp(sin_theta_0, min=1e-8)
  s1 = sin_theta / torch.clamp(sin_theta_0, min=1e-8)
  slerped = s0 * q0 + s1 * q1

  lerped = (1.0 - blend) * q0 + blend * q1
  lerped = lerped / torch.clamp(torch.norm(lerped, dim=-1, keepdim=True), min=1e-8)
  return torch.where(close, lerped, slerped)


class MotionLibrary:
  """Motion library for multi-file motion datasets."""

  def __init__(
    self,
    motion_source: str,
    device: str = "cpu",
    show_progress: bool | None = None,
    required_body_names: tuple[str, ...] | None = None,
  ) -> None:
    self.device = device
    self.required_body_names = required_body_names
    motion_files, per_motion_weights = self._resolve_motion_entries(motion_source)
    if len(motion_files) == 0:
      raise ValueError(f"No motion files found in: {motion_source}")
    self.motion_files = tuple(Path(path) for path in motion_files)

    self.body_names: tuple[str, ...] | None = None
    self._num_dof: int | None = None
    self._num_bodies: int | None = None

    motion_num_frames: list[int] = []
    motion_lengths_s: list[float] = []
    motion_weights: list[float] = []
    joint_pos_list: list[torch.Tensor] = []
    joint_vel_list: list[torch.Tensor] = []
    body_pos_w_list: list[torch.Tensor] = []
    body_quat_w_list: list[torch.Tensor] = []
    body_lin_vel_w_list: list[torch.Tensor] = []
    body_ang_vel_w_list: list[torch.Tensor] = []

    for motion_file, motion_weight in self._iter_motion_entries(
      motion_files=motion_files,
      per_motion_weights=per_motion_weights,
      show_progress=show_progress,
    ):
      payload = load_motion_file(
        motion_file,
        required_body_names=self.required_body_names,
      )
      dt = 1.0 / max(payload.fps, 1e-6)
      selected_body_names = payload.body_names
      joint_pos = torch.tensor(
        payload.joint_pos, dtype=torch.float32, device=self.device
      )
      joint_vel = torch.tensor(
        payload.joint_vel, dtype=torch.float32, device=self.device
      )
      body_pos_w = torch.tensor(
        payload.body_pos_w, dtype=torch.float32, device=self.device
      )
      body_quat_w = torch.tensor(
        payload.body_quat_w, dtype=torch.float32, device=self.device
      )
      body_lin_vel_w = torch.tensor(
        payload.body_lin_vel_w, dtype=torch.float32, device=self.device
      )
      body_ang_vel_w = torch.tensor(
        payload.body_ang_vel_w, dtype=torch.float32, device=self.device
      )

      if joint_pos.ndim != 2:
        raise ValueError(f"Invalid joint_pos shape in {motion_file}: {joint_pos.shape}")
      if joint_vel.shape != joint_pos.shape:
        raise ValueError(
          f"joint_vel must match joint_pos shape in {motion_file}: "
          f"{joint_vel.shape} vs {joint_pos.shape}"
        )
      if body_pos_w.ndim != 3 or body_pos_w.shape[-1] != 3:
        raise ValueError(
          f"Invalid body_pos_w shape in {motion_file}: {body_pos_w.shape}"
        )
      if body_quat_w.ndim != 3 or body_quat_w.shape[-1] != 4:
        raise ValueError(
          f"Invalid body_quat_w shape in {motion_file}: {body_quat_w.shape}"
        )
      if body_lin_vel_w.shape != body_pos_w.shape:
        raise ValueError(
          f"body_lin_vel_w must match body_pos_w shape in {motion_file}: "
          f"{body_lin_vel_w.shape} vs {body_pos_w.shape}"
        )
      if body_ang_vel_w.shape != body_pos_w.shape:
        raise ValueError(
          f"body_ang_vel_w must match body_pos_w shape in {motion_file}: "
          f"{body_ang_vel_w.shape} vs {body_pos_w.shape}"
        )
      if (
        joint_pos.shape[0] != body_pos_w.shape[0]
        or joint_pos.shape[0] != body_quat_w.shape[0]
      ):
        raise ValueError(
          "Frame count mismatch in motion file: "
          f"{motion_file} (joint={joint_pos.shape[0]}, body_pos={body_pos_w.shape[0]}, "
          f"body_quat={body_quat_w.shape[0]})"
        )
      if joint_pos.shape[0] < 2:
        raise ValueError(
          f"Motion {motion_file} has fewer than 2 frames: {joint_pos.shape[0]}"
        )

      if self.body_names is None:
        self.body_names = selected_body_names
      elif self.body_names != selected_body_names:
        raise ValueError(
          "All motion files must share the same selected body name ordering. "
          f"Mismatch in {motion_file}."
        )

      if self._num_dof is None:
        self._num_dof = int(joint_pos.shape[1])
      elif self._num_dof != int(joint_pos.shape[1]):
        raise ValueError(
          "All motion files must share the same number of DoFs. "
          f"Expected {self._num_dof}, got {joint_pos.shape[1]} in {motion_file}."
        )

      if self._num_bodies is None:
        self._num_bodies = int(body_pos_w.shape[1])
      elif self._num_bodies != int(body_pos_w.shape[1]):
        raise ValueError(
          "All motion files must share the same number of bodies. "
          f"Expected {self._num_bodies}, got {body_pos_w.shape[1]} in {motion_file}."
        )

      num_frames = int(joint_pos.shape[0])
      motion_num_frames.append(num_frames)
      motion_lengths_s.append(dt * float(num_frames - 1))
      motion_weights.append(float(motion_weight))
      joint_pos_list.append(joint_pos)
      joint_vel_list.append(joint_vel)
      body_pos_w_list.append(body_pos_w)
      body_quat_w_list.append(body_quat_w)
      body_lin_vel_w_list.append(body_lin_vel_w)
      body_ang_vel_w_list.append(body_ang_vel_w)

    assert self.body_names is not None
    assert self._num_dof is not None
    assert self._num_bodies is not None

    self.motion_num_frames = torch.tensor(
      motion_num_frames, dtype=torch.long, device=self.device
    )
    self.motion_lengths_s = torch.tensor(
      motion_lengths_s, dtype=torch.float32, device=self.device
    )
    self.motion_weights = torch.tensor(
      motion_weights, dtype=torch.float32, device=self.device
    )
    if torch.all(self.motion_weights <= 0):
      self.motion_weights = torch.ones_like(self.motion_weights)
    self.motion_weights = self.motion_weights / self.motion_weights.sum()

    lengths_shifted = self.motion_num_frames.roll(1)
    lengths_shifted[0] = 0
    self.motion_start_idx = lengths_shifted.cumsum(0)

    self.joint_pos = torch.cat(joint_pos_list, dim=0)
    self.joint_vel = torch.cat(joint_vel_list, dim=0)
    self.body_pos_w = torch.cat(body_pos_w_list, dim=0)
    self.body_quat_w = torch.cat(body_quat_w_list, dim=0)
    self.body_lin_vel_w = torch.cat(body_lin_vel_w_list, dim=0)
    self.body_ang_vel_w = torch.cat(body_ang_vel_w_list, dim=0)

  @classmethod
  def _iter_motion_entries(
    cls,
    motion_files: list[Path],
    per_motion_weights: list[float],
    show_progress: bool | None,
  ):
    entries = list(zip(motion_files, per_motion_weights, strict=True))
    if _should_show_progress(show_progress) and tqdm is not None and len(entries) > 1:
      return tqdm(
        entries,
        total=len(entries),
        desc="Loading motions",
        unit="file",
        leave=False,
        dynamic_ncols=True,
      )
    return entries

  @staticmethod
  def _find_motion_files(root: Path) -> list[Path]:
    return sorted(
      path for suffix in ("*.npz", "*.pkl", "*.pickle") for path in root.rglob(suffix)
    )

  @classmethod
  def _resolve_motion_entries(
    cls, motion_source: str
  ) -> tuple[list[Path], list[float]]:
    source = Path(motion_source)
    if source.suffix in (".yaml", ".yml") and source.is_file():
      return cls._resolve_motion_entries_from_yaml(source)
    if source.is_dir():
      files = cls._find_motion_files(source)
      return files, [1.0] * len(files)
    if source.suffix in (".npz", ".pkl", ".pickle") and source.is_file():
      return [source], [1.0]
    raise ValueError(
      "Motion source must be an existing .npz/.pkl/.yaml file or directory. "
      f"Got: {motion_source}"
    )

  @classmethod
  def _resolve_motion_entries_from_yaml(
    cls, yaml_path: Path
  ) -> tuple[list[Path], list[float]]:
    config = _load_yaml_config(yaml_path)
    root_path_raw = config.get("root_path", ".")
    if not isinstance(root_path_raw, str):
      raise ValueError(
        f"`root_path` must be a string in {yaml_path}. Got: {type(root_path_raw)}"
      )
    root_path = Path(root_path_raw)
    if not root_path.is_absolute():
      root_path = (yaml_path.parent / root_path).resolve()

    subfolders = config.get("subfolders")
    if not isinstance(subfolders, list):
      raise ValueError(
        f"`subfolders` must be a list in {yaml_path}. "
        "Expected entries like `{name: cnrs, weight: 1.0}`."
      )

    motion_files: list[Path] = []
    motion_weights: list[float] = []
    for entry in subfolders:
      if not isinstance(entry, dict):
        raise ValueError(f"Invalid subfolder entry in {yaml_path}: {entry}")

      folder_name = entry.get("name", entry.get("folder", entry.get("subfolder")))
      if not isinstance(folder_name, str) or folder_name == "":
        raise ValueError(
          f"Subfolder entry is missing `name` (or `folder`) in {yaml_path}: {entry}"
        )

      weight_raw = entry.get("weight", 1.0)
      try:
        weight = float(weight_raw)
      except (TypeError, ValueError) as exc:
        raise ValueError(
          f"Invalid weight for subfolder `{folder_name}` in {yaml_path}: {weight_raw}"
        ) from exc
      if weight < 0.0:
        raise ValueError(
          f"Weight for subfolder `{folder_name}` must be non-negative in {yaml_path}."
        )

      folder_path = root_path / folder_name
      if not folder_path.exists():
        raise ValueError(
          f"Configured subfolder does not exist: {folder_path} (from {yaml_path})"
        )

      folder_files = cls._find_motion_files(folder_path)
      if len(folder_files) == 0:
        warnings.warn(
          f"No motion files found in configured subfolder: {folder_path}",
          stacklevel=2,
        )
        continue

      motion_files.extend(folder_files)
      motion_weights.extend([weight] * len(folder_files))

    if len(motion_files) == 0:
      raise ValueError(f"No motion files resolved from YAML config: {yaml_path}")
    return motion_files, motion_weights

  def num_motions(self) -> int:
    """Return the number of clips loaded into the library."""
    return int(self.motion_num_frames.shape[0])

  def get_motion_length(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Return clip lengths in seconds for the given motion ids."""
    return self.motion_lengths_s[motion_ids]

  def sample_motions(self, n: int) -> torch.Tensor:
    """Sample motion ids according to per-clip weights."""
    return torch.multinomial(self.motion_weights, num_samples=n, replacement=True)

  def sample_time(self, motion_ids: torch.Tensor) -> torch.Tensor:
    """Sample a random time uniformly within each selected clip."""
    return torch.rand(motion_ids.shape, device=self.device) * self.get_motion_length(
      motion_ids
    )

  def _calc_frame_blend(
    self, motion_ids: torch.Tensor, motion_times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths_s = self.get_motion_length(motion_ids)
    motion_times = torch.clamp(motion_times, min=0.0)
    motion_times = torch.minimum(motion_times, torch.clamp(lengths_s - 1e-6, min=0.0))

    num_frames = self.motion_num_frames[motion_ids]
    phase = motion_times / torch.clamp(lengths_s, min=1e-6)
    phase = torch.clamp(phase, 0.0, 1.0)

    frame_idx0 = (phase * (num_frames - 1).float()).long()
    frame_idx1 = torch.minimum(frame_idx0 + 1, num_frames - 1)
    blend = phase * (num_frames - 1).float() - frame_idx0.float()

    start_idx = self.motion_start_idx[motion_ids]
    frame_idx0 = frame_idx0 + start_idx
    frame_idx1 = frame_idx1 + start_idx
    return frame_idx0, frame_idx1, blend

  def calc_motion_frame(
    self,
    motion_ids: torch.Tensor,
    motion_times: torch.Tensor,
    anchor_body_index: int,
  ) -> MotionFrameBatch:
    """Interpolate and return motion-reference tensors at the requested times."""
    frame_idx0, frame_idx1, blend = self._calc_frame_blend(motion_ids, motion_times)

    joint_pos0 = self.joint_pos[frame_idx0]
    joint_pos1 = self.joint_pos[frame_idx1]
    joint_vel = self.joint_vel[frame_idx0]

    body_pos0 = self.body_pos_w[frame_idx0]
    body_pos1 = self.body_pos_w[frame_idx1]
    body_quat0 = self.body_quat_w[frame_idx0]
    body_quat1 = self.body_quat_w[frame_idx1]
    body_lin_vel = self.body_lin_vel_w[frame_idx0]
    body_ang_vel = self.body_ang_vel_w[frame_idx0]

    blend_joint = blend.unsqueeze(-1)
    blend_body = blend.unsqueeze(-1).unsqueeze(-1)
    joint_pos = (1.0 - blend_joint) * joint_pos0 + blend_joint * joint_pos1
    body_pos_w = (1.0 - blend_body) * body_pos0 + blend_body * body_pos1

    num_bodies = body_pos_w.shape[1]
    if not (0 <= anchor_body_index < num_bodies):
      raise ValueError(
        f"Invalid anchor_body_index={anchor_body_index}. "
        f"Expected in [0, {num_bodies - 1}]."
      )

    body_quat_w = _quat_slerp_batch(
      body_quat0.reshape(-1, 4),
      body_quat1.reshape(-1, 4),
      blend.unsqueeze(-1).expand(-1, num_bodies).reshape(-1),
    ).reshape(motion_ids.shape[0], num_bodies, 4)

    return MotionFrameBatch(
      joint_pos=joint_pos,
      joint_vel=joint_vel,
      body_pos_w=body_pos_w,
      body_quat_w=body_quat_w,
      body_lin_vel_w=body_lin_vel,
      body_ang_vel_w=body_ang_vel,
      anchor_pos_w=body_pos_w[:, anchor_body_index],
      anchor_quat_w=body_quat_w[:, anchor_body_index],
      anchor_lin_vel_w=body_lin_vel[:, anchor_body_index],
      anchor_ang_vel_w=body_ang_vel[:, anchor_body_index],
    )
