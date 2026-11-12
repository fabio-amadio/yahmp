"""MuJoCo Warp utilities for GPU-parallel policy evaluation."""

from __future__ import annotations

import csv
import json
import math
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from mjlab.actuator import DelayedActuatorCfg
from mjlab.entity import Entity, EntityArticulationInfoCfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.sensor import ContactMatch, ContactSensor, ContactSensorCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import quat_error_magnitude

from yahmp.mdp.motion.base import MotionCommand
from yahmp.mdp.motion.sampling import reset_robot_to_reference

FALL_CONTACT_SENSOR = "torso_ground_contact"
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


@dataclass(frozen=True)
class GpuTrackingEvalConfig:
  task_id: str
  output_dir: str
  key_body_names: tuple[str, ...]
  device: str = "cuda:0"
  num_envs: int = 256


def _disable_actuator_delays(env_cfg) -> None:
  robot_cfg = env_cfg.scene.entities["robot"]
  articulation = robot_cfg.articulation
  if articulation is None:
    return
  actuators = tuple(
    actuator.base_cfg if isinstance(actuator, DelayedActuatorCfg) else actuator
    for actuator in articulation.actuators
  )
  robot_cfg.articulation = EntityArticulationInfoCfg(
    actuators=actuators,
    soft_joint_pos_limit_factor=articulation.soft_joint_pos_limit_factor,
  )


def build_gpu_eval_env(cfg: GpuTrackingEvalConfig) -> ManagerBasedRlEnv:
  """Build a deterministic vectorized environment on MuJoCo Warp."""
  if not cfg.device.startswith("cuda"):
    raise ValueError("GPU evaluation requires a CUDA device, e.g. `cuda:0`.")
  if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available to PyTorch.")
  if cfg.num_envs <= 0:
    raise ValueError(f"`num_envs` must be positive, got {cfg.num_envs}.")

  env_cfg = load_env_cfg(cfg.task_id, play=True)
  env_cfg.scene.num_envs = int(cfg.num_envs)
  env_cfg.seed = 0
  env_cfg.episode_length_s = 1.0e9
  env_cfg.events = {}
  env_cfg.rewards = {}
  env_cfg.terminations = {}
  env_cfg.curriculum = {}
  env_cfg.metrics = {}

  for group in env_cfg.observations.values():
    if group is not None:
      group.enable_corruption = False

  motion_cfg = env_cfg.commands["motion"]
  motion_cfg.sampling_mode = "start"
  motion_cfg.pose_range = {}
  motion_cfg.velocity_range = {}
  motion_cfg.joint_position_range = (0.0, 0.0)

  sensors = tuple(
    sensor for sensor in env_cfg.scene.sensors if sensor.name != FALL_CONTACT_SENSOR
  )
  torso_ground_contact = ContactSensorCfg(
    name=FALL_CONTACT_SENSOR,
    primary=ContactMatch(mode="body", pattern="torso_link", entity="robot"),
    secondary=ContactMatch(mode="body", pattern="terrain"),
    fields=("found",),
    reduce="maxforce",
    num_slots=1,
    secondary_policy="any",
  )
  env_cfg.scene.sensors = (*sensors, torso_ground_contact)

  _disable_actuator_delays(env_cfg)
  return ManagerBasedRlEnv(cfg=env_cfg, device=cfg.device, render_mode=None)


def _motion_command(env: ManagerBasedRlEnv) -> MotionCommand:
  command = env.command_manager.get_term("motion")
  if not isinstance(command, MotionCommand):
    raise TypeError(f"Expected a MotionCommand, got {type(command)}.")
  if not command._uses_motion_library or command.motion_lib is None:
    raise ValueError("GPU dataset evaluation requires a multi-motion source.")
  return command


def _fall_contact_mask(env: ManagerBasedRlEnv) -> torch.Tensor:
  sensor = env.scene[FALL_CONTACT_SENSOR]
  if not isinstance(sensor, ContactSensor) or sensor.data.found is None:
    raise RuntimeError("Torso-ground contact sensor is unavailable.")
  return (sensor.data.found > 0).reshape(env.num_envs, -1).any(dim=-1)


def _assign_motion_batch(
  env: ManagerBasedRlEnv,
  motion_indices: list[int],
) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
  """Reset all environments and assign deterministic clips from time zero."""
  if not motion_indices:
    raise ValueError("Cannot assign an empty motion batch.")
  if len(motion_indices) > env.num_envs:
    raise ValueError("Motion batch is larger than the number of environments.")

  env.reset(seed=env.cfg.seed)
  command = _motion_command(env)
  env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
  padded = motion_indices + [motion_indices[-1]] * (env.num_envs - len(motion_indices))
  command.motion_ids[:] = torch.tensor(padded, device=env.device, dtype=torch.long)
  command.motion_time_offsets.zero_()
  command.time_steps.zero_()
  command.time_left.fill_(1.0e9)
  command._refresh_motion_frame()
  reset_robot_to_reference(command, env_ids)

  env.action_manager.reset(env_ids)
  env.observation_manager.reset(env_ids)
  env.episode_length_buf.zero_()
  env.scene.write_data_to_sim()
  env.sim.forward()
  env.sim.sense()
  observations = env.observation_manager.compute(update_history=True)
  return observations, _fall_contact_mask(env)


def _step_gpu_env(
  env: ManagerBasedRlEnv,
  actions: torch.Tensor,
) -> tuple[dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
  """Step all environments and retain actuator forces at every physics step."""
  robot = env.scene["robot"]
  if not isinstance(robot, Entity):
    raise TypeError("Scene entity `robot` is not an Entity.")

  env.action_manager.process_action(actions.to(env.device))
  torque_abs_sum = torch.zeros(
    (env.num_envs, robot.num_actuators), device=env.device, dtype=torch.float32
  )
  torque_abs_max = torch.zeros_like(torque_abs_sum)

  for _ in range(env.cfg.decimation):
    env._sim_step_counter += 1
    env.action_manager.apply_action()
    env.scene.write_data_to_sim()
    env.sim.step()
    env.scene.update(dt=env.physics_dt)
    torque_abs = robot.data.actuator_force.abs().to(torch.float32)
    torque_abs_sum += torque_abs
    torque_abs_max = torch.maximum(torque_abs_max, torque_abs)

  env.episode_length_buf += 1
  env.common_step_counter += 1
  env.sim.forward()
  env.command_manager.compute(dt=env.step_dt)
  env.sim.sense()
  observations = env.observation_manager.compute(update_history=True)
  return observations, torque_abs_sum, torque_abs_max


def _tracking_metrics(
  command: MotionCommand,
  key_body_indices: torch.Tensor,
) -> dict[str, torch.Tensor]:
  return {
    "error_anchor_pos": torch.norm(
      command.anchor_pos_w - command.robot_anchor_pos_w, dim=-1
    ),
    "error_anchor_rot": quat_error_magnitude(
      command.anchor_quat_w, command.robot_anchor_quat_w
    ),
    "error_anchor_lin_vel": torch.norm(
      command.anchor_lin_vel_w - command.robot_anchor_lin_vel_w, dim=-1
    ),
    "error_anchor_ang_vel": torch.norm(
      command.anchor_ang_vel_w - command.robot_anchor_ang_vel_w, dim=-1
    ),
    "error_body_pos": torch.norm(
      command.body_pos_relative_w[:, key_body_indices]
      - command.robot_body_pos_w[:, key_body_indices],
      dim=-1,
    ).mean(dim=-1),
    "error_body_rot": quat_error_magnitude(
      command.body_quat_relative_w[:, key_body_indices],
      command.robot_body_quat_w[:, key_body_indices],
    ).mean(dim=-1),
    "error_body_lin_vel": torch.norm(
      command.body_lin_vel_w[:, key_body_indices]
      - command.robot_body_lin_vel_w[:, key_body_indices],
      dim=-1,
    ).mean(dim=-1),
    "error_body_ang_vel": torch.norm(
      command.body_ang_vel_w[:, key_body_indices]
      - command.robot_body_ang_vel_w[:, key_body_indices],
      dim=-1,
    ).mean(dim=-1),
    "error_joint_pos": torch.norm(command.joint_pos - command.robot_joint_pos, dim=-1),
    "error_joint_vel": torch.norm(command.joint_vel - command.robot_joint_vel, dim=-1),
  }


def _finite_mean(rows: list[dict[str, Any]], key: str) -> float:
  values = [float(row[key]) for row in rows if math.isfinite(float(row[key]))]
  return sum(values) / len(values) if values else math.nan


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
  valid = [row for row in rows if bool(row["valid_for_success_eval"])]
  successes = sum(int(bool(row["success"])) for row in valid)
  return {
    "num_clips_total": len(rows),
    "num_clips_valid_for_success_eval": len(valid),
    "num_reference_ground_contact_start": len(rows) - len(valid),
    "success_count": successes,
    "failure_count": len(valid) - successes,
    "success_rate": successes / len(valid) if valid else math.nan,
    "completion_ratio_macro_mean": _finite_mean(valid, "completion_ratio"),
    "tracking_metrics_macro_mean": {
      name: _finite_mean(valid, name) for name in TRACKING_METRIC_NAMES
    },
    "actuation_macro_mean": {
      name: _finite_mean(valid, name)
      for name in (
        "avg_abs_torque_all",
        "max_abs_torque_all",
        "avg_abs_torque_upper",
        "max_abs_torque_upper",
        "avg_abs_torque_lower",
        "max_abs_torque_lower",
      )
    },
  }


def _actuator_group_masks(
  actuator_names: tuple[str, ...], device: str
) -> tuple[torch.Tensor, ...]:
  lower_keywords = ("hip_", "knee_", "ankle_")
  lower = torch.tensor(
    [any(keyword in name for keyword in lower_keywords) for name in actuator_names],
    dtype=torch.bool,
    device=device,
  )
  return ~lower, lower


def run_gpu_tracking_evaluation(
  cfg: GpuTrackingEvalConfig,
  env: ManagerBasedRlEnv,
  policy: Callable[[dict[str, torch.Tensor]], torch.Tensor],
  *,
  policy_metadata: dict[str, Any],
) -> dict[str, Any]:
  """Evaluate all configured motions in fixed-size GPU batches."""
  started = time.perf_counter()
  command = _motion_command(env)
  motion_lib = command.motion_lib
  assert motion_lib is not None
  motion_files = tuple(Path(path) for path in motion_lib.motion_files)
  num_motions = len(motion_files)
  if num_motions == 0:
    raise ValueError("The configured motion source is empty.")

  missing = [name for name in cfg.key_body_names if name not in command.cfg.body_names]
  if missing:
    raise ValueError(f"Key bodies are not tracked by the command: {missing}")
  key_indices = torch.tensor(
    [command.cfg.body_names.index(name) for name in cfg.key_body_names],
    device=env.device,
    dtype=torch.long,
  )

  robot = env.scene["robot"]
  if not isinstance(robot, Entity):
    raise TypeError("Scene entity `robot` is not an Entity.")
  actuator_names = tuple(robot.actuator_names)
  upper_mask, lower_mask = _actuator_group_masks(actuator_names, env.device)
  rows: list[dict[str, Any]] = []

  for chunk_start in range(0, num_motions, env.num_envs):
    chunk = list(range(chunk_start, min(chunk_start + env.num_envs, num_motions)))
    valid_count = len(chunk)
    observations, reference_starts_in_contact = _assign_motion_batch(env, chunk)

    motion_ids = torch.tensor(chunk, device=env.device, dtype=torch.long)
    durations = motion_lib.get_motion_length(motion_ids).to(torch.float32)
    num_frames = motion_lib.motion_num_frames[motion_ids]
    total_steps = torch.ceil(durations / float(env.step_dt)).to(torch.long)
    padded_total_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    padded_total_steps[:valid_count] = total_steps
    batch_steps = int(total_steps.max().item()) if valid_count else 0

    failed = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
    completed_steps = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    metric_sums = {
      name: torch.zeros(env.num_envs, device=env.device, dtype=torch.float64)
      for name in TRACKING_METRIC_NAMES
    }
    metric_counts = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    torque_sum = torch.zeros(
      (env.num_envs, robot.num_actuators), device=env.device, dtype=torch.float64
    )
    torque_max = torch.zeros_like(torque_sum)
    torque_samples = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    valid_slots = torch.arange(env.num_envs, device=env.device) < valid_count

    print(
      f"[INFO] GPU batch {chunk[0]}-{chunk[-1]} "
      f"({valid_count} motions, batch_steps={batch_steps})"
    )

    for step_idx in range(batch_steps):
      active = (
        valid_slots
        & (step_idx < padded_total_steps)
        & ~failed
        & ~reference_starts_in_contact
      )
      if not torch.any(active):
        break

      actions = policy(observations)
      if actions.shape != (env.num_envs, env.action_manager.total_action_dim):
        raise ValueError(
          f"Expected actions {(env.num_envs, env.action_manager.total_action_dim)}, "
          f"got {tuple(actions.shape)}."
        )
      actions = torch.where(active[:, None], actions, torch.zeros_like(actions))
      observations, step_torque_sum, step_torque_max = _step_gpu_env(env, actions)

      torque_sum[active] += step_torque_sum[active].to(torch.float64)
      torque_max[active] = torch.maximum(
        torque_max[active], step_torque_max[active].to(torch.float64)
      )
      torque_samples[active] += env.cfg.decimation

      fall_contact = _fall_contact_mask(env)
      failed |= active & fall_contact
      survived = active & ~fall_contact

      metrics = _tracking_metrics(command, key_indices)
      for name, values in metrics.items():
        metric_sums[name][survived] += values[survived].to(torch.float64)
      metric_counts[survived] += 1
      completed_steps[survived] = step_idx + 1

    counts = metric_counts[:valid_count].clamp_min(1).to(torch.float64)
    torque_count = torque_samples[:valid_count].clamp_min(1).to(torch.float64)
    metric_means = {
      name: (values[:valid_count] / counts).cpu().tolist()
      for name, values in metric_sums.items()
    }
    torque_avg_joint = torque_sum[:valid_count] / torque_count[:, None]
    torque_values = {
      "avg_abs_torque_all": torque_avg_joint.mean(dim=-1),
      "max_abs_torque_all": torque_max[:valid_count].max(dim=-1).values,
      "avg_abs_torque_upper": torque_avg_joint[:, upper_mask].mean(dim=-1),
      "max_abs_torque_upper": torque_max[:valid_count, upper_mask].max(dim=-1).values,
      "avg_abs_torque_lower": torque_avg_joint[:, lower_mask].mean(dim=-1),
      "max_abs_torque_lower": torque_max[:valid_count, lower_mask].max(dim=-1).values,
    }
    torque_values_cpu = {
      name: values.cpu().tolist() for name, values in torque_values.items()
    }

    failed_cpu = failed[:valid_count].cpu().tolist()
    contact_cpu = reference_starts_in_contact[:valid_count].cpu().tolist()
    completed_cpu = completed_steps[:valid_count].cpu().tolist()
    total_cpu = total_steps.cpu().tolist()
    num_frames_cpu = num_frames.cpu().tolist()
    duration_cpu = durations.cpu().tolist()
    for local_idx, motion_idx in enumerate(chunk):
      valid = not bool(contact_cpu[local_idx])
      completion = (
        float(completed_cpu[local_idx]) / float(total_cpu[local_idx])
        if total_cpu[local_idx] > 0
        else 1.0
      )
      row: dict[str, Any] = {
        "motion_index": motion_idx,
        "motion_file": str(motion_files[motion_idx]),
        "num_frames": int(num_frames_cpu[local_idx]),
        "total_steps": int(total_cpu[local_idx]),
        "completed_steps": int(completed_cpu[local_idx]),
        "valid_for_success_eval": valid,
        "reference_ground_contact_start": bool(contact_cpu[local_idx]),
        "success": bool(
          valid
          and not failed_cpu[local_idx]
          and completed_cpu[local_idx] >= total_cpu[local_idx]
        ),
        "failed": bool(failed_cpu[local_idx]),
        "completion_ratio": completion,
        "duration_s": float(duration_cpu[local_idx]),
        "completed_duration_s": min(
          float(completed_cpu[local_idx]) * float(env.step_dt),
          float(duration_cpu[local_idx]),
        ),
      }
      row.update({name: values[local_idx] for name, values in metric_means.items()})
      row.update(
        {name: values[local_idx] for name, values in torque_values_cpu.items()}
      )
      rows.append(row)

  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = output_dir / "per_motion_success.csv"
  with csv_path.open("w", newline="", encoding="utf-8") as file:
    writer = csv.DictWriter(file, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

  summary = _summary(rows)
  summary.update(
    {
      "device": cfg.device,
      "num_envs": env.num_envs,
      "physics_dt": float(env.physics_dt),
      "control_dt": float(env.step_dt),
      "wall_time_s": time.perf_counter() - started,
      "policy": policy_metadata,
      "motion_source": str(command.cfg.motion_file),
    }
  )
  summary_path = output_dir / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  print(json.dumps(summary, indent=2))
  return {
    "per_motion_csv": str(csv_path),
    "summary_json": str(summary_path),
    "summary": summary,
  }
