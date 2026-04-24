"""Run an original TWIST2 ONNX policy on a selected motion in MuJoCo."""

from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import (
  MotionClip,
  MotionFrame,
  _build_task_scene,
  _configure_camera,
  _create_onnx_session,
  _joint_addresses,
  _quat_rotate_inverse,
  _root_addresses,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ONNX_PATH = REPO_ROOT / "assets/models/twist2_1017_25k.onnx"
DEFAULT_MJLAB_TASK_ID = "Mjlab-YAHMP-Unitree-G1"

TWIST2_G1_JOINT_NAMES = (
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

TWIST2_DEFAULT_DOF_POS = np.array(
  [
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.4,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.4,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
  ],
  dtype=np.float64,
)

TWIST2_SIM2SIM_RESET_DOF_POS = np.array(
  [
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
    -0.2,
    0.0,
    0.0,
    0.4,
    -0.2,
    0.0,
    0.0,
    0.0,
    0.0,
    0.0,
    0.2,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
    0.0,
    -0.2,
    0.0,
    1.2,
    0.0,
    0.0,
    0.0,
  ],
  dtype=np.float64,
)

ANKLE_DOF_INDICES = (4, 5, 10, 11)
N_MIMIC_OBS = 35
N_PROPRIO_OBS = 92
N_OBS_SINGLE = N_MIMIC_OBS + N_PROPRIO_OBS
HISTORY_LEN = 10
OBS_DIM_NO_FUTURE = N_OBS_SINGLE * (HISTORY_LEN + 1)
OBS_DIM_WITH_CURRENT_AS_FUTURE = OBS_DIM_NO_FUTURE + N_MIMIC_OBS


@dataclass(frozen=True)
class Twist2ControlProfile:
  name: str
  stiffness: np.ndarray
  damping: np.ndarray
  torque_limits: np.ndarray
  action_scale: np.ndarray


def _full(value: float) -> np.ndarray:
  return np.full(29, float(value), dtype=np.float64)


def _profile_sim2sim() -> Twist2ControlProfile:
  stiffness = np.array(
    [
      100,
      100,
      100,
      150,
      40,
      40,
      100,
      100,
      100,
      150,
      40,
      40,
      150,
      150,
      150,
      40,
      40,
      40,
      40,
      4.0,
      4.0,
      4.0,
      40,
      40,
      40,
      40,
      4.0,
      4.0,
      4.0,
    ],
    dtype=np.float64,
  )
  damping = np.array(
    [
      2,
      2,
      2,
      4,
      2,
      2,
      2,
      2,
      2,
      4,
      2,
      2,
      4,
      4,
      4,
      5,
      5,
      5,
      5,
      0.2,
      0.2,
      0.2,
      5,
      5,
      5,
      5,
      0.2,
      0.2,
      0.2,
    ],
    dtype=np.float64,
  )
  return Twist2ControlProfile(
    name="twist2-sim2sim",
    stiffness=stiffness,
    damping=damping,
    torque_limits=stiffness.copy(),
    action_scale=_full(0.5),
  )


def _profile_real_yaml() -> Twist2ControlProfile:
  stiffness = np.array(
    [
      100,
      100,
      100,
      150,
      40,
      40,
      100,
      100,
      100,
      150,
      40,
      40,
      150,
      150,
      150,
      40,
      40,
      40,
      40,
      20,
      20,
      20,
      40,
      40,
      40,
      40,
      20,
      20,
      20,
    ],
    dtype=np.float64,
  )
  damping = np.array(
    [
      2,
      2,
      2,
      4,
      2,
      2,
      2,
      2,
      2,
      4,
      2,
      2,
      4,
      4,
      4,
      5,
      5,
      5,
      5,
      1,
      1,
      1,
      5,
      5,
      5,
      5,
      1,
      1,
      1,
    ],
    dtype=np.float64,
  )
  return Twist2ControlProfile(
    name="twist2-real-yaml",
    stiffness=stiffness,
    damping=damping,
    torque_limits=stiffness.copy(),
    action_scale=_full(0.5),
  )


def _profile_training() -> Twist2ControlProfile:
  stiffness = np.array(
    [
      100,
      100,
      100,
      150,
      40,
      40,
      100,
      100,
      100,
      150,
      40,
      40,
      150,
      150,
      150,
      40,
      40,
      40,
      40,
      40,
      40,
      40,
      40,
      40,
      40,
      40,
      40,
      40,
      40,
    ],
    dtype=np.float64,
  )
  damping = np.array(
    [
      2,
      2,
      2,
      4,
      2,
      2,
      2,
      2,
      2,
      4,
      2,
      2,
      4,
      4,
      4,
      5,
      5,
      5,
      5,
      5,
      5,
      5,
      5,
      5,
      5,
      5,
      5,
      5,
      5,
    ],
    dtype=np.float64,
  )
  return Twist2ControlProfile(
    name="twist2-training",
    stiffness=stiffness,
    damping=damping,
    torque_limits=stiffness.copy(),
    action_scale=_full(0.5),
  )


CONTROL_PROFILES = {
  "twist2-sim2sim": _profile_sim2sim,
  "twist2-real-yaml": _profile_real_yaml,
  "twist2-training": _profile_training,
}


def _onnx_input_dim(session: Any) -> int:
  shape = session.get_inputs()[0].shape
  if len(shape) != 2:
    raise ValueError(f"Expected a rank-2 ONNX input, got shape {shape}.")
  dim = shape[1]
  if isinstance(dim, str) or dim is None:
    raise ValueError(
      f"Could not infer static ONNX observation dimension from input shape {shape}."
    )
  return int(dim)


def _actuator_ids_for_joint_names(
  model: mujoco.MjModel, joint_names: tuple[str, ...]
) -> np.ndarray:
  joint_to_actuator: dict[str, int] = {}
  for actuator_id in range(model.nu):
    joint_id = int(model.actuator_trnid[actuator_id, 0])
    if joint_id < 0:
      continue
    joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
    if joint_name is None:
      continue
    joint_to_actuator[joint_name] = actuator_id
    joint_to_actuator[joint_name.split("/")[-1]] = actuator_id
  return np.asarray([joint_to_actuator[name] for name in joint_names], dtype=np.int32)


def _override_position_actuator_gains(
  model: mujoco.MjModel,
  actuator_ids: np.ndarray,
  control_profile: Twist2ControlProfile,
) -> None:
  """Rewrite mjlab position actuators to TWIST2's PD gains/torque limits."""
  for joint_idx, actuator_id in enumerate(actuator_ids):
    kp = float(control_profile.stiffness[joint_idx])
    kd = float(control_profile.damping[joint_idx])
    limit = float(control_profile.torque_limits[joint_idx])

    model.actuator_gainprm[actuator_id, :] = 0.0
    model.actuator_gainprm[actuator_id, 0] = kp
    model.actuator_biasprm[actuator_id, :] = 0.0
    model.actuator_biasprm[actuator_id, 1] = -kp
    model.actuator_biasprm[actuator_id, 2] = -kd
    model.actuator_forcerange[actuator_id, :] = (-limit, limit)
    model.actuator_forcelimited[actuator_id] = 1

    # The policy outputs position targets around the TWIST2 default pose. Keep
    # target clipping out of MuJoCo so the control profile is applied directly.
    model.actuator_ctrlrange[actuator_id, :] = (-1.0e6, 1.0e6)
    model.actuator_ctrllimited[actuator_id] = 0


def _quat_to_euler_wxyz(q: np.ndarray) -> np.ndarray:
  q = np.asarray(q, dtype=np.float64)
  q = q / max(np.linalg.norm(q), 1.0e-12)
  w, x, y, z = q
  roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
  sinp = 2.0 * (w * y - z * x)
  pitch = np.arcsin(np.clip(sinp, -1.0, 1.0))
  yaw = np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
  return np.array((roll, pitch, yaw), dtype=np.float64)


def _twist2_mimic_command(frame: MotionFrame) -> np.ndarray:
  rpy = _quat_to_euler_wxyz(frame.root_quat_w)
  root_lin_vel_b = _quat_rotate_inverse(frame.root_quat_w, frame.root_lin_vel_w)
  root_ang_vel_b = _quat_rotate_inverse(frame.root_quat_w, frame.root_ang_vel_w)
  return np.concatenate(
    (
      root_lin_vel_b[:2],
      frame.root_pos_w[2:3],
      rpy[:2],
      root_ang_vel_b[2:3],
      frame.joint_pos,
    )
  ).astype(np.float32)


def _extract_robot_state(
  data: mujoco.MjData,
  joint_qpos_adr: np.ndarray,
  joint_qvel_adr: np.ndarray,
  root_qpos_adr: int,
  root_qvel_adr: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  dof_pos = np.asarray(data.qpos[joint_qpos_adr], dtype=np.float64).copy()
  dof_vel = np.asarray(data.qvel[joint_qvel_adr], dtype=np.float64).copy()
  quat = np.asarray(data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7], dtype=np.float64)
  ang_vel = np.asarray(
    data.qvel[root_qvel_adr + 3 : root_qvel_adr + 6], dtype=np.float64
  )
  return dof_pos, dof_vel, quat.copy(), ang_vel.copy()


def _twist2_proprio(
  data: mujoco.MjData,
  joint_qpos_adr: np.ndarray,
  joint_qvel_adr: np.ndarray,
  root_qpos_adr: int,
  root_qvel_adr: int,
  last_action: np.ndarray,
  *,
  zero_ankle_vel: bool,
) -> np.ndarray:
  dof_pos, dof_vel, quat, ang_vel = _extract_robot_state(
    data, joint_qpos_adr, joint_qvel_adr, root_qpos_adr, root_qvel_adr
  )
  rpy = _quat_to_euler_wxyz(quat)
  obs_dof_vel = dof_vel.copy()
  if zero_ankle_vel:
    obs_dof_vel[list(ANKLE_DOF_INDICES)] = 0.0
  return np.concatenate(
    (
      ang_vel * 0.25,
      rpy[:2],
      dof_pos - TWIST2_DEFAULT_DOF_POS,
      obs_dof_vel * 0.05,
      last_action,
    )
  ).astype(np.float32)


def _build_observation(
  *,
  data: mujoco.MjData,
  frame: MotionFrame,
  history: deque[np.ndarray],
  last_action: np.ndarray,
  joint_qpos_adr: np.ndarray,
  joint_qvel_adr: np.ndarray,
  root_qpos_adr: int,
  root_qvel_adr: int,
  include_future_block: bool,
  zero_ankle_vel: bool,
) -> tuple[np.ndarray, np.ndarray]:
  mimic = _twist2_mimic_command(frame)
  proprio = _twist2_proprio(
    data,
    joint_qpos_adr,
    joint_qvel_adr,
    root_qpos_adr,
    root_qvel_adr,
    last_action,
    zero_ankle_vel=zero_ankle_vel,
  )
  current = np.concatenate((mimic, proprio)).astype(np.float32)
  obs_parts = [current, np.asarray(history, dtype=np.float32).reshape(-1)]
  if include_future_block:
    # Original TWIST2 exports a future branch, but the sim deploy fills it with
    # the current mimic command because the configured future offset is [0].
    obs_parts.append(mimic)
  return np.concatenate(obs_parts).astype(np.float32), current


def _set_reference_state(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  frame: MotionFrame,
  joint_qpos_adr: np.ndarray,
  joint_qvel_adr: np.ndarray,
  root_qpos_adr: int,
  root_qvel_adr: int,
) -> None:
  mujoco.mj_resetData(model, data)
  data.qpos[root_qpos_adr : root_qpos_adr + 3] = frame.root_pos_w
  data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = frame.root_quat_w
  data.qvel[root_qvel_adr : root_qvel_adr + 3] = frame.root_lin_vel_w
  data.qvel[root_qvel_adr + 3 : root_qvel_adr + 6] = frame.root_ang_vel_w
  data.qpos[joint_qpos_adr] = frame.joint_pos
  data.qvel[joint_qvel_adr] = frame.joint_vel
  mujoco.mj_forward(model, data)


def _set_twist2_default_state(
  model: mujoco.MjModel,
  data: mujoco.MjData,
  joint_qpos_adr: np.ndarray,
  joint_qvel_adr: np.ndarray,
  root_qpos_adr: int,
  root_qvel_adr: int,
) -> None:
  del joint_qvel_adr, root_qvel_adr
  mujoco.mj_resetData(model, data)
  data.qpos[root_qpos_adr : root_qpos_adr + 3] = (0.0, 0.0, 0.793)
  data.qpos[root_qpos_adr + 3 : root_qpos_adr + 7] = (1.0, 0.0, 0.0, 0.0)
  data.qpos[joint_qpos_adr] = TWIST2_SIM2SIM_RESET_DOF_POS
  data.qvel[:] = 0.0
  mujoco.mj_forward(model, data)


def _init_history(
  init_mode: str,
  current: np.ndarray,
) -> deque[np.ndarray]:
  history: deque[np.ndarray] = deque(maxlen=HISTORY_LEN)
  seed = current if init_mode == "current" else np.zeros(N_OBS_SINGLE, dtype=np.float32)
  for _ in range(HISTORY_LEN):
    history.append(seed.astype(np.float32).copy())
  return history


def _tracking_summary(
  data: mujoco.MjData,
  frame: MotionFrame,
  joint_qpos_adr: np.ndarray,
  root_qpos_adr: int,
) -> tuple[float, float, float]:
  dof_pos = np.asarray(data.qpos[joint_qpos_adr], dtype=np.float64)
  root_pos = np.asarray(data.qpos[root_qpos_adr : root_qpos_adr + 3], dtype=np.float64)
  joint_rmse = float(np.sqrt(np.mean(np.square(dof_pos - frame.joint_pos))))
  root_xyz_err = float(np.linalg.norm(root_pos - frame.root_pos_w))
  root_z_err = float(abs(root_pos[2] - frame.root_pos_w[2]))
  return joint_rmse, root_xyz_err, root_z_err


def _print_setup(
  *,
  onnx_path: Path,
  task_id: str,
  motion_file: Path,
  clip: MotionClip,
  providers: list[str],
  input_dim: int,
  include_future_block: bool,
  control_profile: Twist2ControlProfile,
  physics_dt: float,
  control_dt: float,
  control_decimation: int,
  init_pose: str,
  history_init: str,
  zero_ankle_vel: bool,
) -> None:
  print(f"[INFO] ONNX: {onnx_path}")
  print(f"[INFO] mjlab task: {task_id}")
  print(f"[INFO] Motion file: {motion_file}")
  print(f"[INFO] ONNX Runtime providers: {providers}")
  print(f"[INFO] ONNX input dim: {input_dim}")
  print(f"[INFO] Future/current-as-future block: {include_future_block}")
  print(f"[INFO] Control profile: {control_profile.name}")
  print(f"[INFO] MuJoCo dt: physics={physics_dt:.4f} s control={control_dt:.4f} s")
  print(f"[INFO] Control decimation: {control_decimation}")
  print(f"[INFO] Init pose: {init_pose}")
  print(f"[INFO] History init: {history_init}")
  print(f"[INFO] Zero ankle velocity obs: {zero_ankle_vel}")
  print(
    f"[INFO] Motion clip: {clip.num_frames} frames @ {clip.fps:.2f} FPS "
    f"({clip.length_s:.2f}s)"
  )
  print(
    "[INFO] Wrist gains/action scale: "
    f"kp={control_profile.stiffness[19]:.3g}, "
    f"kd={control_profile.damping[19]:.3g}, "
    f"scale={control_profile.action_scale[19]:.3g}"
  )


def run(
  *,
  onnx_path: Path,
  motion_file: Path,
  ort_provider: str,
  control_profile_name: str,
  init_pose: str,
  history_init: str,
  zero_ankle_vel: bool,
  clip_actions: float,
  loop: bool,
  headless: bool,
  max_time_s: float | None,
  print_interval_s: float,
  real_time: bool,
  root_body_name: str,
) -> None:
  session, providers = _create_onnx_session(onnx_path, ort_provider)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name
  input_dim = _onnx_input_dim(session)
  if input_dim == OBS_DIM_WITH_CURRENT_AS_FUTURE:
    include_future_block = True
  elif input_dim == OBS_DIM_NO_FUTURE:
    include_future_block = False
  else:
    raise ValueError(
      "Unsupported TWIST2 ONNX input dimension. Expected "
      f"{OBS_DIM_WITH_CURRENT_AS_FUTURE} with current-as-future or "
      f"{OBS_DIM_NO_FUTURE} without future, got {input_dim}."
    )

  control_profile = CONTROL_PROFILES[control_profile_name]()
  clip = MotionClip(motion_file, root_body_name=root_body_name)
  task_id = DEFAULT_MJLAB_TASK_ID
  model, viewer_cfg, scene_physics_dt, scene_control_dt = _build_task_scene(task_id)
  data = mujoco.MjData(model)
  if model.nu != 29:
    raise ValueError(f"Expected 29 MuJoCo actuators for TWIST2 G1, got {model.nu}.")

  joint_qpos_adr, joint_qvel_adr = _joint_addresses(model, TWIST2_G1_JOINT_NAMES)
  root_qpos_adr, root_qvel_adr = _root_addresses(model, root_body_name)
  actuator_ids = _actuator_ids_for_joint_names(model, TWIST2_G1_JOINT_NAMES)
  _override_position_actuator_gains(model, actuator_ids, control_profile)

  sim_dt_value = float(scene_physics_dt)
  control_dt = float(scene_control_dt)
  control_decimation = int(round(control_dt / sim_dt_value))
  if control_decimation <= 0:
    raise ValueError("Invalid control/physics dt combination.")
  if not np.isclose(
    control_dt, control_decimation * sim_dt_value, rtol=0.0, atol=1.0e-6
  ):
    raise ValueError(
      f"Task scene control_dt={control_dt}s and physics_dt={sim_dt_value}s do not "
      f"produce an integer decimation."
    )

  def reset_rollout() -> tuple[float, np.ndarray, np.ndarray, deque[np.ndarray]]:
    frame0 = clip.sample(0.0)
    if init_pose == "reference":
      _set_reference_state(
        model,
        data,
        frame0,
        joint_qpos_adr,
        joint_qvel_adr,
        root_qpos_adr,
        root_qvel_adr,
      )
    else:
      _set_twist2_default_state(
        model, data, joint_qpos_adr, joint_qvel_adr, root_qpos_adr, root_qvel_adr
      )
    last_action = np.zeros(29, dtype=np.float32)
    initial_obs, initial_current = _build_observation(
      data=data,
      frame=frame0,
      history=deque([np.zeros(N_OBS_SINGLE, dtype=np.float32)] * HISTORY_LEN),
      last_action=last_action,
      joint_qpos_adr=joint_qpos_adr,
      joint_qvel_adr=joint_qvel_adr,
      root_qpos_adr=root_qpos_adr,
      root_qvel_adr=root_qvel_adr,
      include_future_block=include_future_block,
      zero_ankle_vel=zero_ankle_vel,
    )
    del initial_obs
    history = _init_history(history_init, initial_current)
    pd_target = np.asarray(data.qpos[joint_qpos_adr], dtype=np.float64).copy()
    return 0.0, last_action, pd_target, history

  playback_time_s, last_action, pd_target, history = reset_rollout()
  _print_setup(
    onnx_path=onnx_path,
    task_id=task_id,
    motion_file=motion_file,
    clip=clip,
    providers=providers,
    input_dim=input_dim,
    include_future_block=include_future_block,
    control_profile=control_profile,
    physics_dt=sim_dt_value,
    control_dt=control_dt,
    control_decimation=control_decimation,
    init_pose=init_pose,
    history_init=history_init,
    zero_ankle_vel=zero_ankle_vel,
  )

  stop_time_s = max_time_s
  if stop_time_s is None and headless:
    stop_time_s = clip.length_s
  next_print_s = 0.0

  def policy_step() -> None:
    nonlocal playback_time_s, last_action, pd_target, history, next_print_s
    if playback_time_s >= clip.length_s:
      if loop:
        playback_time_s, last_action, pd_target, history = reset_rollout()
      else:
        raise StopIteration

    frame = clip.sample(playback_time_s)
    obs, current = _build_observation(
      data=data,
      frame=frame,
      history=history,
      last_action=last_action,
      joint_qpos_adr=joint_qpos_adr,
      joint_qvel_adr=joint_qvel_adr,
      root_qpos_adr=root_qpos_adr,
      root_qvel_adr=root_qvel_adr,
      include_future_block=include_future_block,
      zero_ankle_vel=zero_ankle_vel,
    )
    if obs.shape != (input_dim,):
      raise ValueError(f"Built obs shape {obs.shape}, expected ({input_dim},).")
    raw_action = (
      session.run(
        [output_name], {input_name: obs[None, :].astype(np.float32, copy=False)}
      )[0]
      .reshape(-1)
      .astype(np.float32)
    )
    if raw_action.shape != (29,):
      raise ValueError(f"Expected 29 policy actions, got {raw_action.shape}.")

    last_action = raw_action.copy()
    action = np.clip(raw_action.astype(np.float64), -clip_actions, clip_actions)
    pd_target = TWIST2_DEFAULT_DOF_POS + action * control_profile.action_scale
    history.append(current)

    if print_interval_s > 0.0 and playback_time_s + 1.0e-9 >= next_print_s:
      joint_rmse, root_xyz_err, root_z_err = _tracking_summary(
        data, frame, joint_qpos_adr, root_qpos_adr
      )
      print(
        "[INFO] "
        f"t={playback_time_s:6.2f}s "
        f"joint_rmse={joint_rmse:7.4f} "
        f"root_xyz={root_xyz_err:7.4f} "
        f"root_z={root_z_err:7.4f} "
        f"action_l2={np.linalg.norm(raw_action):7.3f}"
      )
      next_print_s += print_interval_s

    playback_time_s += control_dt

  def sim_step() -> None:
    data.ctrl[actuator_ids] = pd_target
    mujoco.mj_step(model, data)

  def run_loop(viewer: Any | None = None) -> None:
    while viewer is None or viewer.is_running():
      wall_t0 = time.perf_counter()
      if stop_time_s is not None and playback_time_s >= stop_time_s:
        break
      try:
        policy_step()
      except StopIteration:
        break
      for _ in range(control_decimation):
        sim_step()
      if viewer is not None:
        viewer.sync()
      if real_time:
        elapsed = time.perf_counter() - wall_t0
        time.sleep(max(0.0, control_dt - elapsed))

  if headless:
    run_loop(None)
    return

  import mujoco.viewer as mujoco_viewer

  with mujoco_viewer.launch_passive(model, data) as viewer:
    _configure_camera(viewer, model, root_body_name, viewer_cfg)
    run_loop(viewer)


def _build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Run an original TWIST2 ONNX policy on one reference motion."
  )
  parser.add_argument(
    "--onnx-path",
    type=Path,
    default=DEFAULT_ONNX_PATH,
    help="Path to the original TWIST2 ONNX policy.",
  )
  parser.add_argument(
    "--motion-file",
    "--motion-npz",
    dest="motion_file",
    type=Path,
    required=True,
    metavar="MOTION_FILE",
    help="Reference motion clip in YAHMP/mjlab NPZ or TWIST2 PKL format.",
  )
  parser.add_argument(
    "--ort-provider",
    choices=("auto", "cpu", "cuda"),
    default="auto",
    help="ONNX Runtime provider preference.",
  )
  parser.add_argument(
    "--control-profile",
    choices=tuple(CONTROL_PROFILES.keys()),
    default="twist2-sim2sim",
    help="PD/action-scale profile to use for torque control.",
  )
  parser.add_argument(
    "--init-pose",
    choices=("reference", "twist2-default"),
    default="reference",
    help="Whether to initialize from motion frame 0 or TWIST2's default sim pose.",
  )
  parser.add_argument(
    "--history-init",
    choices=("zeros", "current"),
    default="zeros",
    help="TWIST2 deploy initializes history with zeros; current is useful for debugging.",
  )
  parser.add_argument(
    "--zero-ankle-vel",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Match original TWIST2 deployment by zeroing ankle velocity obs.",
  )
  parser.add_argument(
    "--clip-actions",
    type=float,
    default=10.0,
    help="Raw action clipping before action_scale. Original sim script uses 10.",
  )
  parser.add_argument(
    "--loop",
    action=argparse.BooleanOptionalAction,
    default=False,
    help="Restart the motion when the clip ends.",
  )
  parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without opening a MuJoCo viewer.",
  )
  parser.add_argument(
    "--max-time-s",
    type=float,
    default=None,
    help="Stop after this many reference seconds. Defaults to one clip in headless mode.",
  )
  parser.add_argument(
    "--print-interval-s",
    type=float,
    default=1.0,
    help="Print tracking diagnostics every N reference seconds. Use 0 to disable.",
  )
  parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Sleep to keep the simulation near real time.",
  )
  parser.add_argument(
    "--root-body-name",
    type=str,
    default="pelvis",
    help="Root body name in both XML and motion file.",
  )
  return parser


def main() -> None:
  args = _build_argparser().parse_args()
  run(
    onnx_path=args.onnx_path.expanduser().resolve(),
    motion_file=args.motion_file.expanduser().resolve(),
    ort_provider=str(args.ort_provider),
    control_profile_name=str(args.control_profile),
    init_pose=str(args.init_pose),
    history_init=str(args.history_init),
    zero_ankle_vel=bool(args.zero_ankle_vel),
    clip_actions=float(args.clip_actions),
    loop=bool(args.loop),
    headless=bool(args.headless),
    max_time_s=args.max_time_s,
    print_interval_s=float(args.print_interval_s),
    real_time=bool(args.real_time),
    root_body_name=str(args.root_body_name),
  )


if __name__ == "__main__":
  main()
