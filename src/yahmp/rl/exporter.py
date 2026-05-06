import json

import numpy as np
import onnx
import torch
from mjlab.entity import Entity
from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import JointPositionAction

from yahmp.mdp.actions import (
  JointPositionVelocityAction,
  ResidualJointPositionAction,
  ResidualJointPositionVelocityAction,
)
from yahmp.mdp.motion.base import MotionCommand


def _list_to_csv(arr, *, decimals: int = 3, delimiter: str = ",") -> str:
  del decimals
  values: list[str] = []
  for x in arr:
    if isinstance(x, bool):
      values.append(str(x))
    elif isinstance(x, (int, np.integer)):
      values.append(str(int(x)))
    elif isinstance(x, (float, np.floating)):
      values.append(format(float(x), ".17g"))
    else:
      values.append(str(x))
  return delimiter.join(values)


def _export_value(value):
  if value is None:
    return None
  if isinstance(value, torch.Tensor):
    value = value.detach().cpu()
    if value.ndim == 0:
      return value.item()
    if value.ndim > 1:
      value = value[0]
    return value.tolist()
  return value


def _serialize_metadata_value(value) -> str:
  if isinstance(value, torch.Tensor):
    value = _export_value(value)
  if isinstance(value, (list, tuple)):
    if all(isinstance(item, (str, int, float, bool)) for item in value):
      return _list_to_csv(list(value))
    return json.dumps(value)
  if isinstance(value, dict):
    return json.dumps(value)
  return str(value)


def _get_action_semantics(action_term) -> str:
  if isinstance(action_term, ResidualJointPositionVelocityAction):
    return "residual_joint_position_with_vel_ref"
  if isinstance(action_term, JointPositionVelocityAction):
    return "joint_position_with_vel_ref"
  if isinstance(action_term, ResidualJointPositionAction):
    return "residual_joint_position"
  if isinstance(action_term, JointPositionAction):
    return "joint_position"
  return "custom"


def _get_action_term_metadata(
  term_name: str, action_term, start: int, end: int
) -> dict:
  action_cfg = getattr(action_term, "cfg", None)
  metadata = {
    "name": term_name,
    "term_class": type(action_term).__name__,
    "cfg_class": type(action_cfg).__name__ if action_cfg is not None else "",
    "semantics": _get_action_semantics(action_term),
    "slice": [start, end],
    "dim": int(action_term.action_dim),
    "target_names": list(getattr(action_term, "target_names", [])),
  }

  action_scale = getattr(action_term, "scale", None)
  if action_scale is not None:
    metadata["scale"] = _export_value(action_scale)

  action_offset = getattr(action_term, "offset", None)
  if action_offset is not None:
    metadata["offset"] = _export_value(action_offset)

  if action_cfg is not None and hasattr(action_cfg, "use_default_offset"):
    metadata["use_default_offset"] = bool(action_cfg.use_default_offset)
  if action_cfg is not None and hasattr(action_cfg, "command_name"):
    metadata["reference_command_name"] = str(action_cfg.command_name)
  if action_cfg is not None and hasattr(action_cfg, "vel_ref_alpha"):
    metadata["vel_ref_alpha"] = float(action_cfg.vel_ref_alpha)
  if action_cfg is not None and hasattr(action_cfg, "vel_ref_eta"):
    metadata["vel_ref_eta"] = float(action_cfg.vel_ref_eta)
  if action_cfg is not None and hasattr(action_cfg, "vel_ref_dt"):
    vel_ref_dt = (
      action_cfg.vel_ref_dt
      if action_cfg.vel_ref_dt is not None
      else action_term._env.step_dt
    )
    metadata["vel_ref_dt"] = float(vel_ref_dt)

  return metadata


def _get_action_metadata(
  env: ManagerBasedRlEnv,
) -> dict[str, list | str | float | int | bool]:
  action_manager = env.action_manager
  if len(action_manager.active_terms) != 1:
    raise ValueError(
      "ONNX metadata export currently supports a single action term for deployment."
    )

  term_name = action_manager.active_terms[0]
  action_term = action_manager.get_term(term_name)
  primary = _get_action_term_metadata(
    term_name, action_term, 0, int(action_term.action_dim)
  )

  metadata: dict[str, list | str | float | int | bool] = {
    "action_semantics": primary["semantics"],
    "action_dim": primary["dim"],
    "action_target_names": primary["target_names"],
  }
  if "scale" in primary:
    metadata["action_scale"] = primary["scale"]
  if "offset" in primary:
    metadata["action_offset"] = primary["offset"]
  return metadata


def _get_observation_metadata(
  env: ManagerBasedRlEnv,
) -> dict[str, list | str | float | int | bool]:
  observation_manager = env.observation_manager
  group_name = "actor"
  observation_terms = observation_manager.active_terms[group_name]
  term_dims = observation_manager._group_obs_term_dim[group_name]
  term_cfgs = observation_manager._group_obs_term_cfgs[group_name]
  group_dim = observation_manager.group_obs_dim[group_name]

  terms_layout = []
  offset = 0
  for name, dims, term_cfg in zip(
    observation_terms, term_dims, term_cfgs, strict=False
  ):
    flat_dim = int(np.prod(dims))
    terms_layout.append(
      {
        "name": name,
        "shape": list(dims),
        "flat_dim": flat_dim,
        "slice": [offset, offset + flat_dim],
        "history_length": int(term_cfg.history_length),
        "flatten_history_dim": bool(term_cfg.flatten_history_dim),
      }
    )
    offset += flat_dim

  return {
    "observation_group": group_name,
    "observation_dim": list(group_dim) if isinstance(group_dim, tuple) else group_dim,
    "observation_terms_layout": terms_layout,
  }


def _get_export_observation_group_name(env: ManagerBasedRlEnv) -> str:
  del env
  return "actor"


def _get_command_observation_term_cfg(env: ManagerBasedRlEnv):
  observation_manager = env.observation_manager
  group_name = _get_export_observation_group_name(env)
  observation_terms = observation_manager.active_terms[group_name]
  term_cfgs = observation_manager._group_obs_term_cfgs[group_name]

  for name, term_cfg in zip(observation_terms, term_cfgs, strict=False):
    if name == "command":
      return term_cfg
  return None


def _infer_motion_representation_name(env: ManagerBasedRlEnv) -> str:
  command_term_cfg = _get_command_observation_term_cfg(env)
  if command_term_cfg is None:
    return "default"

  obs_func = getattr(command_term_cfg, "func", None)
  func_name = getattr(obs_func, "__name__", "")
  if func_name in ("generated_commands", "motion_student_command"):
    return "default"
  if func_name == "motion_teacher_command":
    return "teacher"
  if func_name == "motion_command_representation":
    return str(
      getattr(command_term_cfg, "params", {}).get("representation_name", "default")
    )
  return "default"


def _get_motion_metadata(
  env: ManagerBasedRlEnv,
) -> dict[str, list | str | float | int | bool]:
  motion_term = env.command_manager.get_term("motion")
  assert isinstance(motion_term, MotionCommand)

  representation_name = _infer_motion_representation_name(env)
  if not motion_term.has_command_representation(representation_name):
    raise KeyError(
      f"Motion command {type(motion_term).__name__} does not define representation "
      f"{representation_name!r}."
    )

  representation = motion_term.get_command_representation(representation_name)
  if representation_name == "teacher":
    step_offsets = tuple(motion_term.future_sampling_step_offsets)
  elif len(motion_term.command_representation_names) > 1:
    step_offsets = ()
  else:
    step_offsets = tuple(motion_term.future_sampling_step_offsets)

  return {
    "motion_command_class": type(motion_term).__name__,
    "motion_command_cfg_class": type(motion_term.cfg).__name__,
    "motion_command_representation_name": representation_name,
    "motion_command_dim": int(representation.shape[-1]),
    "motion_command_step_offsets": list(step_offsets),
    "anchor_body_name": motion_term.cfg.anchor_body_name,
    "body_names": list(getattr(motion_term.cfg, "body_names", ())),
    "root_body_name": motion_term.root_body_name,
    "left_hand_body_name": getattr(motion_term.cfg, "left_hand_body_name", ""),
    "right_hand_body_name": getattr(motion_term.cfg, "right_hand_body_name", ""),
  }


def _get_base_metadata(
  env: ManagerBasedRlEnv, run_path: str
) -> dict[str, list | str | float | int | bool]:
  robot: Entity = env.scene["robot"]

  joint_name_to_ctrl_id = {}
  for actuator in robot.spec.actuators:
    joint_name = actuator.target.split("/")[-1]
    joint_name_to_ctrl_id[joint_name] = actuator.id
  ctrl_ids_natural = [
    joint_name_to_ctrl_id[jname]
    for jname in robot.joint_names
    if jname in joint_name_to_ctrl_id
  ]

  joint_stiffness = env.sim.mj_model.actuator_gainprm[ctrl_ids_natural, 0]
  joint_damping = -env.sim.mj_model.actuator_biasprm[ctrl_ids_natural, 2]
  metadata: dict[str, list | str | float | int | bool] = {
    "physics_dt": float(env.physics_dt),
    "control_dt": float(env.step_dt),
    "joint_names": list(robot.joint_names),
    "joint_stiffness": joint_stiffness.tolist(),
    "joint_damping": joint_damping.tolist(),
    "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
  }
  metadata.update(_get_action_metadata(env))
  metadata.update(_get_observation_metadata(env))
  return metadata


def attach_onnx_metadata(env: ManagerBasedRlEnv, run_path: str, onnx_path: str) -> None:
  model = onnx.load(onnx_path)
  metadata = _get_base_metadata(env, run_path)
  metadata.update(_get_motion_metadata(env))

  existing = {entry.key: entry.value for entry in model.metadata_props}
  del model.metadata_props[:]
  for key, value in metadata.items():
    existing[key] = _serialize_metadata_value(value)

  for key, value in existing.items():
    entry = onnx.StringStringEntryProto()
    entry.key = key
    entry.value = value
    model.metadata_props.append(entry)

  onnx.save(model, onnx_path)
