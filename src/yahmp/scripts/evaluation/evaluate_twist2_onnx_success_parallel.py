"""Evaluate original TWIST2 ONNX success over a motion source in batches."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import mjlab
import mujoco
import numpy as np
import tyro

from yahmp.config.g1.env_cfgs import G1_COMPARISON_KEY_BODY_NAMES
from yahmp.scripts.deploy.run_twist2_onnx_mujoco import (
  CONTROL_PROFILES,
  OBS_DIM_NO_FUTURE,
  OBS_DIM_WITH_CURRENT_AS_FUTURE,
  TWIST2_DEFAULT_DOF_POS,
  TWIST2_G1_JOINT_NAMES,
  _actuator_ids_for_joint_names,
  _build_observation,
  _create_onnx_session,
  _init_history,
  _onnx_input_dim,
  _override_position_actuator_gains,
  _set_reference_state,
)
from yahmp.scripts.deploy.run_twist2_onnx_mujoco import (
  DEFAULT_MJLAB_TASK_ID as DEFAULT_TWIST2_TASK_ID,
)
from yahmp.scripts.deploy.run_twist2_onnx_mujoco import (
  DEFAULT_ONNX_PATH as DEFAULT_TWIST2_ONNX_PATH,
)
from yahmp.scripts.deploy.run_twist2_onnx_mujoco import (
  HISTORY_LEN as TWIST2_HISTORY_LEN,
)
from yahmp.scripts.deploy.run_twist2_onnx_mujoco import (
  N_OBS_SINGLE as TWIST2_N_OBS_SINGLE,
)
from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import (
  MotionClip,
  _build_task_scene,
  _joint_addresses,
  _root_addresses,
)
from yahmp.scripts.evaluation.tracking_eval_utils import (
  DEFAULT_GROUND_FAIL_BODY_NAMES,
  DEFAULT_HAND_PUSH_BODY_NAMES,
  TRACKING_METRIC_NAMES,
  HandPushMirror,
  HandPushMirrorConfig,
  actuation_stats_to_row,
  compute_tracking_metrics,
  ground_contact_active,
  init_actuation_stats,
  resolve_body_ids,
  resolve_motion_files,
  resolve_motion_source,
  safe_motion_time,
  sample_clip_body_positions_w,
  update_actuation_stats,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets/twist2_onnx_success_eval"


@dataclass(frozen=True)
class EvaluateTwist2OnnxSuccessParallelConfig:
  task_id: str = DEFAULT_TWIST2_TASK_ID
  onnx_path: str = str(DEFAULT_TWIST2_ONNX_PATH)
  control_profile: str = "twist2-sim2sim"
  motion_source: str | None = None
  output_dir: str = str(DEFAULT_OUTPUT_DIR)
  ort_provider: str = "cpu"
  num_envs: int = 256
  max_motions: int | None = None
  start_motion_index: int = 0
  policy_frequency: float = 50.0
  sim_dt: float | None = None
  history_init: Literal["zeros", "current"] = "zeros"
  zero_ankle_vel: bool = True
  clip_actions: float = 10.0
  ground_fail_body_names: tuple[str, ...] = DEFAULT_GROUND_FAIL_BODY_NAMES
  skip_reference_ground_contact_start: bool = True
  key_body_names: tuple[str, ...] = G1_COMPARISON_KEY_BODY_NAMES
  enable_hand_pushes: bool = False
  hand_push_body_names: tuple[str, ...] = DEFAULT_HAND_PUSH_BODY_NAMES
  hand_push_seed: int = 0
  resume: bool = False
  flush_csv_every_chunk: bool = True


def _finite_mean(values: list[float]) -> float:
  finite = [value for value in values if math.isfinite(value)]
  return sum(finite) / float(len(finite)) if finite else math.nan


def _summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
  total = len(rows)
  valid_rows = [row for row in rows if bool(row["valid_for_success_eval"])]
  valid = len(valid_rows)
  successes = sum(int(bool(row["success"])) for row in valid_rows)
  failures = valid - successes

  def mean(key: str) -> float:
    return _finite_mean([float(row[key]) for row in valid_rows])

  def weighted_completion() -> float:
    num = sum(
      float(row["completion_ratio"]) * float(row["duration_s"]) for row in valid_rows
    )
    den = sum(float(row["duration_s"]) for row in valid_rows)
    return num / den if den > 0.0 else math.nan

  return {
    "num_clips_total": total,
    "num_clips_valid_for_success_eval": valid,
    "num_reference_ground_contact_start": total - valid,
    "success_count": successes,
    "failure_count": failures,
    "success_rate": successes / float(valid) if valid > 0 else math.nan,
    "failure_rate": failures / float(valid) if valid > 0 else math.nan,
    "completion_ratio_macro_mean": mean("completion_ratio"),
    "completion_ratio_duration_weighted_mean": weighted_completion(),
    "tracking_metrics_macro_mean": {
      metric_name: mean(metric_name) for metric_name in TRACKING_METRIC_NAMES
    },
  }


def _build_summary(
  *,
  cfg: EvaluateTwist2OnnxSuccessParallelConfig,
  rt: dict[str, Any],
  motion_source: str,
  rows: list[dict[str, Any]],
  wall_time_s: float,
) -> dict[str, Any]:
  core = _summarize_rows(rows)
  valid_rows = [row for row in rows if bool(row["valid_for_success_eval"])]

  actuation = {
    "avg_abs_torque_all_macro_mean": _finite_mean(
      [float(row["avg_abs_torque_all"]) for row in valid_rows]
    ),
    "max_abs_torque_all_macro_mean": _finite_mean(
      [float(row["max_abs_torque_all"]) for row in valid_rows]
    ),
    "avg_abs_torque_upper_macro_mean": _finite_mean(
      [float(row["avg_abs_torque_upper"]) for row in valid_rows]
    ),
    "max_abs_torque_upper_macro_mean": _finite_mean(
      [float(row["max_abs_torque_upper"]) for row in valid_rows]
    ),
    "avg_abs_torque_lower_macro_mean": _finite_mean(
      [float(row["avg_abs_torque_lower"]) for row in valid_rows]
    ),
    "max_abs_torque_lower_macro_mean": _finite_mean(
      [float(row["max_abs_torque_lower"]) for row in valid_rows]
    ),
  }

  summary = {
    "run": {
      "task_id": cfg.task_id,
      "onnx_path": str(rt["onnx_path"]),
      "motion_source": motion_source,
      "wall_time_s": wall_time_s,
    },
    "simulation": {
      "sim_dt": float(rt["sim_dt"]),
      "control_dt": float(rt["control_dt"]),
      "control_decimation": int(rt["control_decimation"]),
      "policy_frequency_hz": float(cfg.policy_frequency),
      "onnx_runtime_providers": list(rt["providers"]),
      "control_profile": cfg.control_profile,
    },
    "evaluation": {
      "num_clips_total": core["num_clips_total"],
      "num_clips_valid_for_success_eval": core["num_clips_valid_for_success_eval"],
      "num_reference_ground_contact_start": core["num_reference_ground_contact_start"],
      "skip_reference_ground_contact_start": cfg.skip_reference_ground_contact_start,
      "ground_fail_body_names": list(cfg.ground_fail_body_names),
      "resume": cfg.resume,
      "flush_csv_every_chunk": cfg.flush_csv_every_chunk,
    },
    "results": {
      "success_count": core["success_count"],
      "failure_count": core["failure_count"],
      "success_rate": core["success_rate"],
      "failure_rate": core["failure_rate"],
      "completion_ratio_macro_mean": core["completion_ratio_macro_mean"],
      "completion_ratio_duration_weighted_mean": core[
        "completion_ratio_duration_weighted_mean"
      ],
    },
    "tracking_metrics": core["tracking_metrics_macro_mean"],
    "actuation": actuation,
    "disturbance": {
      "enable_hand_pushes": cfg.enable_hand_pushes,
      "hand_push_body_names": list(cfg.hand_push_body_names),
      "hand_push_seed": int(cfg.hand_push_seed),
    },
  }
  if cfg.enable_hand_pushes:
    summary["disturbance"].update(
      {
        "hand_push_mean_disp_m_macro_mean": _finite_mean(
          [float(row["hand_push_mean_disp_m"]) for row in valid_rows]
        ),
        "hand_push_max_disp_m_macro_mean": _finite_mean(
          [float(row["hand_push_max_disp_m"]) for row in valid_rows]
        ),
      }
    )
  return summary


def _build_manifest(
  *,
  output_dir: Path,
  csv_path: Path,
  summary_path: Path,
  rows: list[dict[str, Any]],
) -> dict[str, Any]:
  fieldnames = list(rows[0].keys()) if rows else []
  return {
    "files": {
      "output_dir": str(output_dir),
      "per_motion_csv": str(csv_path),
      "summary_json": str(summary_path),
    },
    "row_count": len(rows),
    "csv_columns": fieldnames,
  }


def _load_existing_rows(
  csv_path: Path,
) -> tuple[list[dict[str, Any]], set[int], list[str] | None]:
  if not csv_path.exists():
    return [], set(), None
  with csv_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)
    fieldnames = list(reader.fieldnames) if reader.fieldnames is not None else None
  processed_indices = {
    int(row["motion_index"])
    for row in rows
    if "motion_index" in row and str(row["motion_index"]).strip() != ""
  }
  return rows, processed_indices, fieldnames


def _append_rows(
  csv_path: Path,
  fieldnames: list[str],
  rows: list[dict[str, Any]],
  *,
  write_header: bool,
) -> None:
  if not rows:
    return
  with csv_path.open("a", encoding="utf-8", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if write_header:
      writer.writeheader()
    writer.writerows(rows)


def _build_model_and_runtime(cfg: EvaluateTwist2OnnxSuccessParallelConfig):
  if cfg.control_profile not in CONTROL_PROFILES:
    raise ValueError(
      f"Unknown TWIST2 control profile `{cfg.control_profile}`. "
      f"Available: {tuple(CONTROL_PROFILES)}"
    )

  onnx_path = Path(cfg.onnx_path).expanduser().resolve()
  session, providers = _create_onnx_session(onnx_path, cfg.ort_provider)
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
      f"{OBS_DIM_WITH_CURRENT_AS_FUTURE} or {OBS_DIM_NO_FUTURE}, got {input_dim}."
    )

  model, _, scene_sim_dt, scene_control_dt = _build_task_scene(cfg.task_id)
  if cfg.sim_dt is not None:
    model.opt.timestep = float(cfg.sim_dt)
    sim_dt = float(model.opt.timestep)
    control_decimation = int(round(1.0 / (float(cfg.policy_frequency) * sim_dt)))
    if control_decimation <= 0:
      raise ValueError("Invalid policy_frequency / sim_dt combination.")
    control_dt = control_decimation * sim_dt
    if not np.isclose(
      control_dt, 1.0 / float(cfg.policy_frequency), rtol=0.0, atol=1.0e-6
    ):
      raise ValueError(
        f"policy_frequency={cfg.policy_frequency}Hz and sim_dt={sim_dt}s do not "
        "produce an integer decimation."
      )
  else:
    sim_dt = float(scene_sim_dt)
    control_dt = float(scene_control_dt)
    control_decimation = int(round(control_dt / sim_dt))

  control_profile = CONTROL_PROFILES[cfg.control_profile]()
  joint_qpos_adr, joint_qvel_adr = _joint_addresses(model, TWIST2_G1_JOINT_NAMES)
  root_qpos_adr, root_qvel_adr = _root_addresses(model, "pelvis")
  actuator_ids = _actuator_ids_for_joint_names(model, TWIST2_G1_JOINT_NAMES)
  _override_position_actuator_gains(model, actuator_ids, control_profile)
  ground_fail_body_ids = resolve_body_ids(model, cfg.ground_fail_body_names)
  hand_push_body_ids = resolve_body_ids(model, cfg.hand_push_body_names)

  return {
    "onnx_path": onnx_path,
    "session": session,
    "providers": providers,
    "input_name": input_name,
    "output_name": output_name,
    "input_dim": input_dim,
    "include_future_block": include_future_block,
    "model": model,
    "sim_dt": sim_dt,
    "control_dt": control_dt,
    "control_decimation": control_decimation,
    "control_profile": control_profile,
    "joint_qpos_adr": joint_qpos_adr,
    "joint_qvel_adr": joint_qvel_adr,
    "root_qpos_adr": root_qpos_adr,
    "root_qvel_adr": root_qvel_adr,
    "actuator_ids": actuator_ids,
    "ground_fail_body_ids": ground_fail_body_ids,
    "hand_push_body_ids": hand_push_body_ids,
  }


def run(cfg: EvaluateTwist2OnnxSuccessParallelConfig) -> dict[str, Any]:
  wall_time_start = time.perf_counter()
  import mjlab.tasks  # noqa: F401

  import yahmp.config.g1  # noqa: F401

  motion_source = resolve_motion_source(cfg.task_id, cfg.motion_source)
  motion_files = resolve_motion_files(motion_source)
  output_dir = Path(cfg.output_dir).expanduser().resolve()
  output_dir.mkdir(parents=True, exist_ok=True)
  csv_path = output_dir / "per_motion_success.csv"
  start = max(int(cfg.start_motion_index), 0)
  stop = len(motion_files)
  if cfg.max_motions is not None:
    stop = min(stop, start + max(int(cfg.max_motions), 0))
  selected_indices = list(range(start, stop))
  rows: list[dict[str, Any]] = []
  csv_fieldnames: list[str] | None = None
  csv_initialized = False
  if cfg.resume:
    rows, processed_indices, csv_fieldnames = _load_existing_rows(csv_path)
    csv_initialized = bool(rows)
    num_before = len(selected_indices)
    selected_indices = [idx for idx in selected_indices if idx not in processed_indices]
    print(
      f"[INFO] Resume mode: loaded {len(rows)} existing rows and skipped "
      f"{num_before - len(selected_indices)} already processed motions."
    )

  rt = _build_model_and_runtime(cfg)
  model: mujoco.MjModel = rt["model"]
  data_pool = [mujoco.MjData(model) for _ in range(max(int(cfg.num_envs), 1))]

  for chunk_start in range(0, len(selected_indices), len(data_pool)):
    chunk = selected_indices[chunk_start : chunk_start + len(data_pool)]
    clips = [
      MotionClip(motion_files[motion_idx], root_body_name="pelvis")
      for motion_idx in chunk
    ]
    total_steps = np.asarray(
      [
        int(math.ceil(clip.length_s / rt["control_dt"])) if clip.length_s > 0.0 else 0
        for clip in clips
      ],
      dtype=np.int64,
    )
    max_steps = int(total_steps.max()) if total_steps.size > 0 else 0

    histories = []
    last_actions = []
    reference_contact_start = np.zeros(len(clips), dtype=bool)
    failed = np.zeros(len(clips), dtype=bool)
    completed_steps = np.zeros(len(clips), dtype=np.int64)
    metric_sums = {
      metric_name: np.zeros(len(clips), dtype=np.float64)
      for metric_name in TRACKING_METRIC_NAMES
    }
    metric_counts = np.zeros(len(clips), dtype=np.int64)
    actuation_stats = init_actuation_stats(len(clips), TWIST2_G1_JOINT_NAMES)
    hand_push_mirrors: list[HandPushMirror] = []

    for local_idx, clip in enumerate(clips):
      data = data_pool[local_idx]
      frame0 = clip.sample(0.0)
      _set_reference_state(
        model,
        data,
        frame0,
        rt["joint_qpos_adr"],
        rt["joint_qvel_adr"],
        rt["root_qpos_adr"],
        rt["root_qvel_adr"],
      )
      reference_contact_start[local_idx] = ground_contact_active(
        model, data, rt["ground_fail_body_ids"]
      )
      last_action = np.zeros(29, dtype=np.float32)
      _, current = _build_observation(
        data=data,
        frame=frame0,
        history=[np.zeros(TWIST2_N_OBS_SINGLE, dtype=np.float32)] * TWIST2_HISTORY_LEN,
        last_action=last_action,
        joint_qpos_adr=rt["joint_qpos_adr"],
        joint_qvel_adr=rt["joint_qvel_adr"],
        root_qpos_adr=rt["root_qpos_adr"],
        root_qvel_adr=rt["root_qvel_adr"],
        include_future_block=rt["include_future_block"],
        zero_ankle_vel=cfg.zero_ankle_vel,
      )
      histories.append(_init_history(cfg.history_init, current))
      last_actions.append(last_action)
      hand_push_mirrors.append(
        HandPushMirror(
          model=model,
          body_ids=rt["hand_push_body_ids"],
          dt=rt["control_dt"],
          cfg=HandPushMirrorConfig(
            enabled=cfg.enable_hand_pushes,
            body_names=cfg.hand_push_body_names,
            seed=cfg.hand_push_seed,
          ),
          seed=cfg.hand_push_seed + int(chunk[local_idx]),
        )
      )

    print(
      f"[INFO] Evaluating clips {chunk[0]}-{chunk[-1]} "
      f"({len(chunk)} envs, max_steps={max_steps})"
    )

    for step_idx in range(max_steps):
      active = (total_steps > step_idx) & ~failed
      if cfg.skip_reference_ground_contact_start:
        active = active & ~reference_contact_start
      active_indices = np.nonzero(active)[0]
      if active_indices.size == 0:
        break

      obs_batch = []
      current_by_idx: dict[int, np.ndarray] = {}
      for local_idx in active_indices:
        clip = clips[local_idx]
        data = data_pool[local_idx]
        time_s = min(float(step_idx) * rt["control_dt"], clip.length_s)
        frame = clip.sample(safe_motion_time(clip, time_s))
        obs, current = _build_observation(
          data=data,
          frame=frame,
          history=histories[local_idx],
          last_action=last_actions[local_idx],
          joint_qpos_adr=rt["joint_qpos_adr"],
          joint_qvel_adr=rt["joint_qvel_adr"],
          root_qpos_adr=rt["root_qpos_adr"],
          root_qvel_adr=rt["root_qvel_adr"],
          include_future_block=rt["include_future_block"],
          zero_ankle_vel=cfg.zero_ankle_vel,
        )
        if obs.shape != (rt["input_dim"],):
          raise ValueError(
            f"Built TWIST2 obs shape {obs.shape}, expected ({rt['input_dim']},)."
          )
        obs_batch.append(obs)
        current_by_idx[int(local_idx)] = current

      raw_actions = (
        rt["session"]
        .run(
          [rt["output_name"]],
          {
            rt["input_name"]: np.asarray(obs_batch, dtype=np.float32),
          },
        )[0]
        .astype(np.float32)
      )
      if raw_actions.shape != (active_indices.size, 29):
        raise ValueError(
          f"Expected batched TWIST2 actions [N,29], got {raw_actions.shape}."
        )

      for batch_idx, local_idx in enumerate(active_indices):
        data = data_pool[local_idx]
        raw_action = raw_actions[batch_idx]
        last_actions[local_idx] = raw_action.copy()
        action = np.clip(
          raw_action.astype(np.float64),
          -float(cfg.clip_actions),
          float(cfg.clip_actions),
        )
        pd_target = TWIST2_DEFAULT_DOF_POS + action * rt["control_profile"].action_scale
        histories[local_idx].append(current_by_idx[int(local_idx)])
        hand_push_mirrors[local_idx].pre_step(
          data,
          sample_clip_body_positions_w(
            clips[local_idx],
            min(float(step_idx) * rt["control_dt"], clips[local_idx].length_s),
            cfg.hand_push_body_names,
          ),
        )
        for _ in range(rt["control_decimation"]):
          data.ctrl[rt["actuator_ids"]] = pd_target
          mujoco.mj_step(model, data)
          update_actuation_stats(
            actuation_stats,
            int(local_idx),
            np.asarray(data.actuator_force[rt["actuator_ids"]], dtype=np.float64),
          )
        hand_push_mirrors[local_idx].post_step(
          data,
          sample_clip_body_positions_w(
            clips[local_idx],
            min(float(step_idx + 1) * rt["control_dt"], clips[local_idx].length_s),
            cfg.hand_push_body_names,
          ),
        )

      failed_by_eval = []
      for local_idx in active_indices:
        data = data_pool[local_idx]
        failed_by_eval.append(
          ground_contact_active(model, data, rt["ground_fail_body_ids"])
        )
      failed_by_eval = np.asarray(failed_by_eval, dtype=bool)
      newly_failed_indices = active_indices[failed_by_eval]
      if newly_failed_indices.size > 0:
        failed[newly_failed_indices] = True

      survived_indices = active_indices[~failed_by_eval]
      for local_idx in survived_indices:
        clip = clips[local_idx]
        data = data_pool[local_idx]
        metric_time_s = min(float(step_idx + 1) * rt["control_dt"], clip.length_s)
        metrics = compute_tracking_metrics(
          model=model,
          data=data,
          clip=clip,
          time_s=metric_time_s,
          joint_qpos_adr=rt["joint_qpos_adr"],
          joint_qvel_adr=rt["joint_qvel_adr"],
          root_body_name="pelvis",
          key_body_names=cfg.key_body_names,
        )
        for metric_name, value in metrics.items():
          metric_sums[metric_name][local_idx] += value
        metric_counts[local_idx] += 1
        completed_steps[local_idx] = step_idx + 1

    chunk_rows: list[dict[str, Any]] = []
    for local_idx, motion_idx in enumerate(chunk):
      valid = (
        not bool(reference_contact_start[local_idx])
        if cfg.skip_reference_ground_contact_start
        else True
      )
      completion_ratio = (
        float(completed_steps[local_idx]) / float(total_steps[local_idx])
        if total_steps[local_idx] > 0
        else 1.0
      )
      success = bool(
        valid
        and (not failed[local_idx])
        and completed_steps[local_idx] >= total_steps[local_idx]
      )
      row: dict[str, Any] = {
        "motion_index": motion_idx,
        "motion_file": str(motion_files[motion_idx]),
        "num_frames": clips[local_idx].num_frames,
        "duration_s": clips[local_idx].length_s,
        "total_steps": int(total_steps[local_idx]),
        "completed_steps": int(completed_steps[local_idx]),
        "completion_ratio": completion_ratio,
        "success": success,
        "reference_ground_contact_start": bool(reference_contact_start[local_idx]),
        "valid_for_success_eval": bool(valid),
      }
      if cfg.enable_hand_pushes:
        row["hand_push_mean_disp_m"] = hand_push_mirrors[local_idx].mean_displacement()
        row["hand_push_max_disp_m"] = hand_push_mirrors[local_idx].max_displacement()
        row["hand_push_num_episodes"] = int(hand_push_mirrors[local_idx].push_episodes)
      row.update(actuation_stats_to_row(actuation_stats, local_idx))
      for metric_name in TRACKING_METRIC_NAMES:
        row[metric_name] = (
          float(metric_sums[metric_name][local_idx]) / float(metric_counts[local_idx])
          if metric_counts[local_idx] > 0
          else math.nan
        )
      rows.append(row)
      chunk_rows.append(row)

    if chunk_rows and cfg.flush_csv_every_chunk:
      if csv_fieldnames is None:
        csv_fieldnames = list(chunk_rows[0].keys())
      _append_rows(
        csv_path,
        csv_fieldnames,
        chunk_rows,
        write_header=not csv_initialized,
      )
      csv_initialized = True

  fieldnames = csv_fieldnames or (
    list(rows[0].keys()) if rows else ["motion_index", "motion_file"]
  )
  if not cfg.flush_csv_every_chunk:
    with csv_path.open("w", encoding="utf-8", newline="") as f:
      writer = csv.DictWriter(f, fieldnames=fieldnames)
      writer.writeheader()
      writer.writerows(rows)

  wall_time_s = time.perf_counter() - wall_time_start
  summary = _build_summary(
    cfg=cfg,
    rt=rt,
    motion_source=motion_source,
    rows=rows,
    wall_time_s=wall_time_s,
  )
  summary_path = output_dir / "summary.json"
  summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
  manifest = _build_manifest(
    output_dir=output_dir,
    csv_path=csv_path,
    summary_path=summary_path,
    rows=rows,
  )
  manifest_path = output_dir / "manifest.json"
  manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

  return {
    "output_dir": str(output_dir),
    "per_motion_csv": str(csv_path),
    "summary_json": str(summary_path),
    "manifest_json": str(manifest_path),
    "summary": summary,
  }


def main() -> None:
  cfg = tyro.cli(
    EvaluateTwist2OnnxSuccessParallelConfig,
    config=mjlab.TYRO_FLAGS,
  )
  result = run(cfg)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
