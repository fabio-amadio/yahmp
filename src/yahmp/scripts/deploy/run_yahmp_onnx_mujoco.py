"""Run an exported base YAHMP ONNX policy on a selected motion in MuJoCo."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np
import onnx
from mjlab.scene import Scene
from mjlab.tasks.registry import list_tasks, load_env_cfg

from yahmp.mdp.motion.library import load_motion_file


def _csv(value: str, cast=str) -> list:
  if not value:
    return []
  return [cast(part.strip()) for part in value.split(",")]


def _yahmp_task_ids() -> tuple[str, ...]:
  import mjlab.tasks  # noqa: F401

  import yahmp.config.g1  # noqa: F401

  preferred = ("Mjlab-YAHMP-Unitree-G1",)
  available = set(list_tasks())
  return tuple(task_id for task_id in preferred if task_id in available)


def _load_onnx_metadata(onnx_path: Path) -> dict[str, str]:
  model = onnx.load(str(onnx_path))
  return {entry.key: entry.value for entry in model.metadata_props}


def _expand_vector(values: list[float], size: int) -> np.ndarray:
  if not values:
    raise ValueError("Expected at least one value.")
  if len(values) == 1:
    values = values * size
  vector = np.asarray(values, dtype=np.float64)
  if vector.shape != (size,):
    raise ValueError(f"Expected shape ({size},), got {vector.shape}.")
  return vector


@dataclass(frozen=True)
class ObsTerm:
  name: str
  flat_dim: int
  history_length: int

  @property
  def step_dim(self) -> int:
    if self.history_length <= 0:
      return self.flat_dim
    if self.flat_dim % self.history_length != 0:
      raise ValueError(
        f"Observation term `{self.name}` has flat_dim={self.flat_dim} "
        f"and history_length={self.history_length}, which do not divide cleanly."
      )
    return self.flat_dim // self.history_length


@dataclass(frozen=True)
class PolicySpec:
  physics_dt: float
  control_dt: float
  joint_names: tuple[str, ...]
  default_joint_pos: np.ndarray
  action_semantics: str
  action_dim: int
  action_target_names: tuple[str, ...]
  action_scale: np.ndarray
  action_offset: np.ndarray
  observation_dim: int
  observation_terms: tuple[ObsTerm, ...]
  motion_command_class: str
  motion_command_representation: str
  motion_command_dim: int
  body_names: tuple[str, ...]
  root_body_name: str

  @classmethod
  def from_onnx(cls, onnx_path: Path) -> "PolicySpec":
    metadata = _load_onnx_metadata(onnx_path)

    def require(key: str) -> str:
      if key not in metadata:
        raise KeyError(
          f"Missing ONNX metadata key `{key}` in {onnx_path}. "
          "Re-export the policy with the current YAHMP exporter."
        )
      return metadata[key]

    action_dim = int(require("action_dim"))
    obs_terms = tuple(
      ObsTerm(
        name=str(entry["name"]),
        flat_dim=int(entry["flat_dim"]),
        history_length=int(entry["history_length"]),
      )
      for entry in json.loads(require("observation_terms_layout"))
    )

    return cls(
      physics_dt=float(require("physics_dt")),
      control_dt=float(require("control_dt")),
      joint_names=tuple(_csv(require("joint_names"))),
      default_joint_pos=np.asarray(
        _csv(require("default_joint_pos"), float), dtype=np.float64
      ),
      action_semantics=require("action_semantics"),
      action_dim=action_dim,
      action_target_names=tuple(_csv(require("action_target_names"))),
      action_scale=_expand_vector(_csv(require("action_scale"), float), action_dim),
      action_offset=_expand_vector(
        _csv(metadata.get("action_offset", "0.0"), float), action_dim
      ),
      observation_dim=int(_csv(require("observation_dim"), int)[0]),
      observation_terms=obs_terms,
      motion_command_class=require("motion_command_class"),
      motion_command_representation=metadata.get(
        "motion_command_representation_name", "default"
      ),
      motion_command_dim=int(metadata.get("motion_command_dim", "0")),
      body_names=tuple(_csv(metadata.get("body_names", ""))),
      root_body_name=require("root_body_name"),
    )

  def validate(self) -> None:
    if self.motion_command_representation != "default":
      raise ValueError(
        "This script only supports base YAHMP policies that consume the default "
        "motion representation."
      )
    if self.motion_command_class != "JointRefAnchorRpMotionCommand":
      raise NotImplementedError(
        "This script supports only the base YAHMP JointRefAnchorRp command, got "
        f"`{self.motion_command_class}`."
      )
    expected_command_dim = 2 * len(self.joint_names) + 6
    if self.motion_command_dim != expected_command_dim:
      raise ValueError(
        "This script supports only the current-step JointRefAnchorRp command "
        f"with dim = 2 * num_joints + 6. Got motion_command_dim="
        f"{self.motion_command_dim}, num_joints={len(self.joint_names)}."
      )
    if self.action_semantics not in {"residual_joint_position", "joint_position"}:
      raise NotImplementedError(
        "This script supports only `residual_joint_position` and "
        f"`joint_position`, got `{self.action_semantics}`."
      )


@dataclass(frozen=True)
class MotionFrame:
  joint_pos: np.ndarray
  joint_vel: np.ndarray
  root_pos_w: np.ndarray
  root_quat_w: np.ndarray
  root_lin_vel_w: np.ndarray
  root_ang_vel_w: np.ndarray


class MotionClip:
  def __init__(self, motion_file: Path, root_body_name: str) -> None:
    payload = load_motion_file(motion_file)
    self.joint_pos = np.asarray(payload.joint_pos, dtype=np.float64)
    self.joint_vel = np.asarray(payload.joint_vel, dtype=np.float64)
    self.body_pos_w = np.asarray(payload.body_pos_w, dtype=np.float64)
    self.body_quat_w = np.asarray(payload.body_quat_w, dtype=np.float64)
    self.body_lin_vel_w = np.asarray(payload.body_lin_vel_w, dtype=np.float64)
    self.body_ang_vel_w = np.asarray(payload.body_ang_vel_w, dtype=np.float64)
    self.fps = float(payload.fps)
    body_names = payload.body_names

    if root_body_name not in body_names:
      raise ValueError(
        f"Root body `{root_body_name}` not found in motion clip {motion_file}."
      )

    self.body_names = body_names
    self._body_name_to_index = {name: idx for idx, name in enumerate(body_names)}
    self.root_body_index = body_names.index(root_body_name)

  @property
  def num_frames(self) -> int:
    return int(self.joint_pos.shape[0])

  @property
  def length_s(self) -> float:
    return float(max(self.num_frames - 1, 0)) / max(self.fps, 1.0e-6)

  def sample(self, time_s: float) -> MotionFrame:
    idx0, idx1, blend = self._sample_indices(time_s)

    return MotionFrame(
      joint_pos=_lerp(self.joint_pos[idx0], self.joint_pos[idx1], blend),
      joint_vel=_lerp(self.joint_vel[idx0], self.joint_vel[idx1], blend),
      root_pos_w=_lerp(
        self.body_pos_w[idx0, self.root_body_index],
        self.body_pos_w[idx1, self.root_body_index],
        blend,
      ),
      root_quat_w=_quat_slerp(
        self.body_quat_w[idx0, self.root_body_index],
        self.body_quat_w[idx1, self.root_body_index],
        blend,
      ),
      root_lin_vel_w=_lerp(
        self.body_lin_vel_w[idx0, self.root_body_index],
        self.body_lin_vel_w[idx1, self.root_body_index],
        blend,
      ),
      root_ang_vel_w=_lerp(
        self.body_ang_vel_w[idx0, self.root_body_index],
        self.body_ang_vel_w[idx1, self.root_body_index],
        blend,
      ),
    )

  def body_index(self, body_name: str) -> int:
    if body_name not in self._body_name_to_index:
      raise ValueError(f"Body `{body_name}` not found in loaded motion clip.")
    return self._body_name_to_index[body_name]

  def sample_body_pose(
    self, time_s: float, body_index: int
  ) -> tuple[np.ndarray, np.ndarray]:
    idx0, idx1, blend = self._sample_indices(time_s)
    return (
      _lerp(
        self.body_pos_w[idx0, body_index], self.body_pos_w[idx1, body_index], blend
      ),
      _quat_slerp(
        self.body_quat_w[idx0, body_index],
        self.body_quat_w[idx1, body_index],
        blend,
      ),
    )

  def _sample_indices(self, time_s: float) -> tuple[int, int, float]:
    if self.length_s <= 0.0:
      clip_time = 0.0
    else:
      clip_time = float(time_s % self.length_s)

    phase = clip_time * self.fps
    idx0 = int(np.floor(phase))
    idx1 = min(idx0 + 1, self.num_frames - 1)
    blend = float(phase - idx0)
    return idx0, idx1, blend


def _lerp(a: np.ndarray, b: np.ndarray, blend: float) -> np.ndarray:
  return (1.0 - blend) * a + blend * b


def _quat_normalize(q: np.ndarray) -> np.ndarray:
  norm = np.linalg.norm(q)
  return (
    np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64) if norm < 1.0e-12 else q / norm
  )


def _quat_conj(q: np.ndarray) -> np.ndarray:
  return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float64)


def _quat_mul(q0: np.ndarray, q1: np.ndarray) -> np.ndarray:
  w0, x0, y0, z0 = q0
  w1, x1, y1, z1 = q1
  return np.array(
    (
      w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
      w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
      w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
      w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
    ),
    dtype=np.float64,
  )


def _quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
  q = _quat_normalize(q)
  qv = np.array([0.0, v[0], v[1], v[2]], dtype=np.float64)
  return _quat_mul(_quat_mul(_quat_conj(q), qv), q)[1:]


def _quat_to_rotmat(q: np.ndarray) -> np.ndarray:
  q = _quat_normalize(q)
  w, x, y, z = q
  return np.array(
    (
      (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
      (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
      (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
    ),
    dtype=np.float64,
  )


def _quat_roll_pitch_yaw(q: np.ndarray) -> np.ndarray:
  q = _quat_normalize(q)
  w, x, y, z = q
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.asarray((roll, pitch, yaw), dtype=np.float64)


def _quat_slerp(q0: np.ndarray, q1: np.ndarray, blend: float) -> np.ndarray:
  q0 = _quat_normalize(np.asarray(q0, dtype=np.float64))
  q1 = _quat_normalize(np.asarray(q1, dtype=np.float64))
  dot = float(np.dot(q0, q1))
  if dot < 0.0:
    q1 = -q1
    dot = -dot

  if dot > 0.9995:
    return _quat_normalize(_lerp(q0, q1, blend))

  dot = float(np.clip(dot, 0.0, 1.0))
  theta_0 = float(np.arccos(dot))
  theta = theta_0 * blend
  sin_theta_0 = float(np.sin(theta_0))
  s0 = np.sin(theta_0 - theta) / max(sin_theta_0, 1.0e-8)
  s1 = np.sin(theta) / max(sin_theta_0, 1.0e-8)
  return _quat_normalize(s0 * q0 + s1 * q1)


def _create_onnx_session(onnx_path: Path, provider: str):
  try:
    import onnxruntime as ort
  except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
      "onnxruntime is required to run this script. Install it in the project "
      "environment before using this viewer."
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

  return ort.InferenceSession(str(onnx_path), providers=providers), providers


def _build_task_scene(task_id: str) -> tuple[mujoco.MjModel, Any, float, float]:
  env_cfg = load_env_cfg(task_id)
  env_cfg.scene.num_envs = 1
  scene = Scene(env_cfg.scene, device="cpu")
  model = scene.compile()
  env_cfg.sim.mujoco.apply(model)
  return (
    model,
    env_cfg.viewer,
    float(env_cfg.sim.mujoco.timestep),
    float(env_cfg.sim.mujoco.timestep * env_cfg.decimation),
  )


def _resolve_id(model: mujoco.MjModel, obj_type: Any, name: str) -> int:
  direct = mujoco.mj_name2id(model, obj_type, name)
  if direct >= 0:
    return direct

  counts = {
    mujoco.mjtObj.mjOBJ_BODY: model.nbody,
    mujoco.mjtObj.mjOBJ_JOINT: model.njnt,
    mujoco.mjtObj.mjOBJ_ACTUATOR: model.nu,
  }
  suffix = f"/{name}"
  for idx in range(counts.get(obj_type, 0)):
    candidate = mujoco.mj_id2name(model, obj_type, idx)
    if candidate is not None and candidate.endswith(suffix):
      return idx
  raise ValueError(f"`{name}` not found in MuJoCo model.")


def _joint_addresses(
  model: mujoco.MjModel, joint_names: tuple[str, ...]
) -> tuple[np.ndarray, np.ndarray]:
  qpos = []
  qvel = []
  for name in joint_names:
    joint_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    qpos.append(int(model.jnt_qposadr[joint_id]))
    qvel.append(int(model.jnt_dofadr[joint_id]))
  return np.asarray(qpos, dtype=np.int32), np.asarray(qvel, dtype=np.int32)


def _actuator_ids(
  model: mujoco.MjModel, target_joint_names: tuple[str, ...]
) -> np.ndarray:
  joint_to_actuator: dict[str, int] = {}
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
      continue
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name is not None:
      joint_to_actuator[joint_name] = actuator_id
      joint_to_actuator[joint_name.split("/")[-1]] = actuator_id
  return np.asarray(
    [joint_to_actuator[name] for name in target_joint_names], dtype=np.int32
  )


def _root_addresses(model: mujoco.MjModel, root_body_name: str) -> tuple[int, int]:
  body_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
  joint_id = int(model.body_jntadr[body_id])
  if int(model.jnt_type[joint_id]) != int(mujoco.mjtJoint.mjJNT_FREE):
    raise ValueError(f"Root body `{root_body_name}` is not attached by a free joint.")
  return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def _configure_camera(
  viewer: Any,
  model: mujoco.MjModel,
  root_body_name: str,
  viewer_cfg: Any,
) -> None:
  viewer.opt.frame = mujoco.mjtFrame.mjFRAME_WORLD.value

  def set_tracking(body_id: int) -> None:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING.value
    viewer.cam.trackbodyid = body_id
    viewer.cam.fixedcamid = -1

  if viewer_cfg is None or not hasattr(viewer_cfg, "origin_type"):
    set_tracking(1)
  elif viewer_cfg.origin_type == viewer_cfg.OriginType.WORLD:
    viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FREE.value
    viewer.cam.trackbodyid = -1
    viewer.cam.fixedcamid = -1
  elif viewer_cfg.origin_type == viewer_cfg.OriginType.ASSET_ROOT:
    set_tracking(_resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name))
  elif viewer_cfg.origin_type == viewer_cfg.OriginType.ASSET_BODY:
    set_tracking(_resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, viewer_cfg.body_name))
  else:  # AUTO
    set_tracking(1)

  viewer.cam.lookat[:] = np.asarray(getattr(viewer_cfg, "lookat", (0.0, 0.0, 0.0)))
  viewer.cam.distance = float(getattr(viewer_cfg, "distance", 3.0))
  viewer.cam.azimuth = float(getattr(viewer_cfg, "azimuth", 90.0))
  viewer.cam.elevation = float(getattr(viewer_cfg, "elevation", -5.0))
  if not bool(getattr(viewer_cfg, "enable_shadows", True)):
    viewer.user_scn.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0


def _initialize_state(
  data: mujoco.MjData,
  frame: MotionFrame,
  spec: PolicySpec,
  joint_qpos_adr: np.ndarray,
  joint_qvel_adr: np.ndarray,
  root_qpos_adr: int,
  root_qvel_adr: int,
  *,
  init_default_joints: bool,
) -> None:
  data.qpos[:] = 0.0
  data.qvel[:] = 0.0
  data.ctrl[:] = 0.0

  data.qpos[root_qpos_adr : root_qpos_adr + 3] = frame.root_pos_w
  data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = frame.root_quat_w
  data.qvel[root_qvel_adr : root_qvel_adr + 3] = frame.root_lin_vel_w
  data.qvel[root_qvel_adr + 3 : root_qvel_adr + 6] = frame.root_ang_vel_w

  data.qpos[joint_qpos_adr] = (
    spec.default_joint_pos if init_default_joints else frame.joint_pos
  )
  data.qvel[joint_qvel_adr] = 0.0 if init_default_joints else frame.joint_vel


def _root_kinematics(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  root_body_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
  body_id = _resolve_id(model, mujoco.mjtObj.mjOBJ_BODY, root_body_name)
  velocity_local = np.zeros(6, dtype=np.float64)
  mujoco.mj_objectVelocity(
    model,
    data,
    mujoco.mjtObj.mjOBJ_BODY,
    body_id,
    velocity_local,
    1,
  )
  return (
    np.asarray(data.xquat[body_id], dtype=np.float64).copy(),
    velocity_local[3:6].copy(),
    velocity_local[0:3].copy(),
  )


def _command_value(
  spec: PolicySpec,
  clip: MotionClip,
  time_s: float,
  frame: MotionFrame,
) -> np.ndarray:
  del clip, time_s, spec
  anchor_lin_vel_b = _quat_rotate_inverse(frame.root_quat_w, frame.root_lin_vel_w)
  anchor_ang_vel_b = _quat_rotate_inverse(frame.root_quat_w, frame.root_ang_vel_w)
  parts = [
    frame.joint_pos,
    frame.joint_vel,
    anchor_lin_vel_b[:2],
    anchor_ang_vel_b[2:3],
    frame.root_pos_w[2:3],
  ]
  roll, pitch, _ = _quat_roll_pitch_yaw(frame.root_quat_w)
  parts.extend(
    (
      np.asarray([roll], dtype=np.float64),
      np.asarray([pitch], dtype=np.float64),
    )
  )
  return np.concatenate(parts).astype(np.float32)


def _term_values(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  spec: PolicySpec,
  clip: MotionClip,
  time_s: float,
  frame: MotionFrame,
  joint_qpos_adr: np.ndarray,
  joint_qvel_adr: np.ndarray,
  previous_action: np.ndarray,
) -> dict[str, np.ndarray]:
  quat_wxyz, base_lin_vel_b, base_ang_vel_b = _root_kinematics(
    model, data, spec.root_body_name
  )
  gravity_w = np.asarray(model.opt.gravity, dtype=np.float64)
  gravity_w = gravity_w / max(np.linalg.norm(gravity_w), 1.0e-12)

  return {
    "command": _command_value(spec, clip, time_s, frame),
    "base_lin_vel": base_lin_vel_b.astype(np.float32),
    "base_ang_vel": base_ang_vel_b.astype(np.float32),
    "projected_gravity": _quat_rotate_inverse(quat_wxyz, gravity_w).astype(np.float32),
    "joint_pos": (
      np.asarray(data.qpos[joint_qpos_adr]) - spec.default_joint_pos
    ).astype(np.float32),
    "joint_vel": np.asarray(data.qvel[joint_qvel_adr], dtype=np.float64).astype(
      np.float32
    ),
    "actions": previous_action.astype(np.float32),
  }


def _current_observation_block(
  terms: dict[str, np.ndarray],
) -> np.ndarray:
  """Return YAHMP's deployment-ready current block used by its history term."""
  return np.concatenate(
    (
      terms["command"],
      terms["base_ang_vel"],
      terms["projected_gravity"],
      terms["joint_pos"],
      terms["joint_vel"],
      terms["actions"],
    )
  ).astype(np.float32)


def _initialize_history(
  spec: PolicySpec, terms: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
  history: dict[str, np.ndarray] = {}
  for term in spec.observation_terms:
    if term.history_length > 0:
      history[term.name] = np.repeat(
        terms[term.name][None, :], term.history_length, axis=0
      ).astype(np.float32)
    elif term.name == "history" and term.name not in terms:
      current = _current_observation_block(terms)
      if term.flat_dim % current.shape[0] != 0:
        raise ValueError(
          "Cannot initialize YAHMP history term: "
          f"history dim={term.flat_dim}, current block dim={current.shape[0]}."
        )
      history_length = term.flat_dim // current.shape[0]
      history[term.name] = np.repeat(current[None, :], history_length, axis=0).astype(
        np.float32
      )
  return history


def _append_history(
  spec: PolicySpec,
  history: dict[str, np.ndarray],
  terms: dict[str, np.ndarray],
) -> None:
  for term in spec.observation_terms:
    if term.history_length <= 0:
      if term.name == "history" and term.name in history and term.name not in terms:
        history[term.name][:-1] = history[term.name][1:]
        history[term.name][-1] = _current_observation_block(terms)
      continue
    history[term.name][:-1] = history[term.name][1:]
    history[term.name][-1] = terms[term.name]


def _build_observation(
  spec: PolicySpec,
  terms: dict[str, np.ndarray],
  history: dict[str, np.ndarray],
) -> np.ndarray:
  parts = []
  for term in spec.observation_terms:
    parts.append(
      history[term.name].reshape(-1)
      if term.history_length > 0 or term.name in history
      else terms[term.name]
    )
  obs = np.concatenate(parts).astype(np.float32)
  if obs.shape != (spec.observation_dim,):
    raise ValueError(
      f"Constructed observation has shape {obs.shape}, expected ({spec.observation_dim},)."
    )
  return obs


def _apply_action(
  data: mujoco.MjData,
  spec: PolicySpec,
  raw_action: np.ndarray,
  frame: MotionFrame,
  action_actuator_ids: np.ndarray,
  action_target_joint_indices: np.ndarray,
) -> None:
  processed = raw_action.astype(np.float64) * spec.action_scale + spec.action_offset
  if spec.action_semantics == "residual_joint_position":
    targets = frame.joint_pos[action_target_joint_indices] + processed
  else:
    targets = processed
  data.ctrl[action_actuator_ids] = targets


def _print_summary(
  spec: PolicySpec,
  clip: MotionClip,
  onnx_path: Path,
  task_id: str,
  motion_file: Path,
  providers: list[str],
) -> None:
  print(f"[INFO] ONNX: {onnx_path}")
  print(f"[INFO] Task scene: {task_id}")
  print(f"[INFO] Motion file: {motion_file}")
  print(f"[INFO] ONNX Runtime providers: {providers}")
  print(
    f"[INFO] Policy action semantics: {spec.action_semantics} (dim={spec.action_dim})"
  )
  print(
    f"[INFO] Motion command: {spec.motion_command_class} (dim={spec.motion_command_dim})"
  )
  print(
    f"[INFO] Simulation dt: physics={spec.physics_dt:.4f}s control={spec.control_dt:.4f}s"
  )
  print(
    f"[INFO] Motion clip: {clip.num_frames} frames @ {clip.fps:.2f} FPS ({clip.length_s:.2f}s)"
  )
  print(
    "[INFO] Observation terms: "
    + ", ".join(term.name for term in spec.observation_terms)
  )


def run(
  *,
  onnx_path: Path,
  motion_file: Path,
  task_id: str,
  ort_provider: str,
  init_default_joints: bool,
) -> None:
  spec = PolicySpec.from_onnx(onnx_path)
  spec.validate()
  clip = MotionClip(motion_file, root_body_name=spec.root_body_name)
  session, providers = _create_onnx_session(onnx_path, ort_provider)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name

  model, viewer_cfg, scene_physics_dt, scene_control_dt = _build_task_scene(task_id)
  if not np.isclose(scene_physics_dt, spec.physics_dt):
    raise ValueError(
      f"Task scene physics_dt={scene_physics_dt} does not match exported policy "
      f"physics_dt={spec.physics_dt}."
    )
  if not np.isclose(scene_control_dt, spec.control_dt):
    raise ValueError(
      f"Task scene control_dt={scene_control_dt} does not match exported policy "
      f"control_dt={spec.control_dt}."
    )

  data = mujoco.MjData(model)
  joint_qpos_adr, joint_qvel_adr = _joint_addresses(model, spec.joint_names)
  action_actuator_ids = _actuator_ids(model, spec.action_target_names)
  action_target_joint_indices = np.asarray(
    [spec.joint_names.index(name) for name in spec.action_target_names],
    dtype=np.int32,
  )
  root_qpos_adr, root_qvel_adr = _root_addresses(model, spec.root_body_name)

  initial_frame = clip.sample(0.0)
  _initialize_state(
    data,
    initial_frame,
    spec,
    joint_qpos_adr,
    joint_qvel_adr,
    root_qpos_adr,
    root_qvel_adr,
    init_default_joints=init_default_joints,
  )
  mujoco.mj_forward(model, data)

  previous_action = np.zeros(spec.action_dim, dtype=np.float32)
  terms = _term_values(
    model,
    data,
    spec,
    clip,
    0.0,
    initial_frame,
    joint_qpos_adr,
    joint_qvel_adr,
    previous_action,
  )
  history = _initialize_history(spec, terms)

  steps_per_control = int(round(spec.control_dt / spec.physics_dt))
  if steps_per_control <= 0:
    raise ValueError(
      f"Invalid control/physics dt ratio: control_dt={spec.control_dt}, "
      f"physics_dt={spec.physics_dt}."
    )

  _print_summary(spec, clip, onnx_path, task_id, motion_file, providers)

  import mujoco.viewer as mujoco_viewer

  playback_time_s = 0.0
  with mujoco_viewer.launch_passive(model, data) as viewer:
    _configure_camera(viewer, model, spec.root_body_name, viewer_cfg)

    while viewer.is_running():
      wall_t0 = time.perf_counter()

      frame = clip.sample(playback_time_s)
      terms = _term_values(
        model,
        data,
        spec,
        clip,
        playback_time_s,
        frame,
        joint_qpos_adr,
        joint_qvel_adr,
        previous_action,
      )
      obs = _build_observation(spec, terms, history)
      raw_action = (
        session.run(
          [output_name],
          {input_name: obs[None, :].astype(np.float32, copy=False)},
        )[0]
        .reshape(-1)
        .astype(np.float32)
      )

      _apply_action(
        data,
        spec,
        raw_action,
        frame,
        action_actuator_ids,
        action_target_joint_indices,
      )
      for _ in range(steps_per_control):
        mujoco.mj_step(model, data)

      playback_time_s += spec.control_dt
      previous_action = raw_action
      next_terms = _term_values(
        model,
        data,
        spec,
        clip,
        playback_time_s,
        clip.sample(playback_time_s),
        joint_qpos_adr,
        joint_qvel_adr,
        previous_action,
      )
      _append_history(spec, history, next_terms)

      viewer.sync()
      elapsed = time.perf_counter() - wall_t0
      time.sleep(max(0.0, spec.control_dt - elapsed))


def _build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Run a base YAHMP ONNX policy directly in MuJoCo."
  )
  parser.add_argument(
    "--onnx-path", type=Path, required=True, help="Exported base YAHMP ONNX policy."
  )
  parser.add_argument(
    "--motion-file",
    "--motion-npz",
    dest="motion_file",
    type=Path,
    required=True,
    metavar="MOTION_FILE",
    help="Reference motion clip in YAHMP NPZ or TWIST2 PKL format.",
  )
  parser.add_argument(
    "--task-id",
    type=str,
    required=True,
    choices=_yahmp_task_ids(),
    help="Registered base YAHMP mjlab task whose scene should be used.",
  )
  parser.add_argument(
    "--ort-provider",
    choices=("auto", "cpu", "cuda"),
    default="auto",
    help="ONNX Runtime execution provider preference.",
  )
  parser.add_argument(
    "--init-default-joints",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Initialize joints from the exported default pose instead of the first reference frame.",
  )
  return parser


def main() -> None:
  parser = _build_argparser()
  args = parser.parse_args()
  run(
    onnx_path=args.onnx_path.expanduser().resolve(),
    motion_file=args.motion_file.expanduser().resolve(),
    task_id=str(args.task_id),
    ort_provider=str(args.ort_provider),
    init_default_joints=bool(args.init_default_joints),
  )


if __name__ == "__main__":
  main()
