"""Evaluate a base YAHMP ONNX policy over a motion source in batches."""

from __future__ import annotations

import csv
import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mjlab
import mujoco
import numpy as np
import tyro

from yahmp.config.g1.env_cfgs import G1_COMPARISON_KEY_BODY_NAMES
from yahmp.scripts.deploy.run_yahmp_onnx_mujoco import (
  MotionClip,
  PolicySpec,
  _actuator_ids,
  _append_history,
  _apply_action,
  _build_observation,
  _build_task_scene,
  _create_onnx_session,
  _initialize_history,
  _initialize_state,
  _joint_addresses,
  _root_addresses,
  _term_values,
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
DEFAULT_ONNX_PATH = REPO_ROOT / "assets/models/g1_yahmp.onnx"
DEFAULT_OUTPUT_DIR = REPO_ROOT / "assets/logs/yahmp_onnx_success_eval"


@dataclass(frozen=True)
class EvaluateYahmpOnnxSuccessParallelConfig:
  task_id: str = "Mjlab-YAHMP-Unitree-G1"
  onnx_path: str = str(DEFAULT_ONNX_PATH)
  motion_source: str | None = None
  output_dir: str = str(DEFAULT_OUTPUT_DIR)
  ort_provider: str = "cpu"
  num_envs: int = 256
  max_motions: int | None = None
  start_motion_index: int = 0
  max_motion_duration_s: float | None = None
  init_default_joints: bool = False
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
  cfg: EvaluateYahmpOnnxSuccessParallelConfig,
  rt: dict[str, Any],
  spec: PolicySpec,
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
      "physics_dt": float(spec.physics_dt),
      "control_dt": float(spec.control_dt),
      "control_decimation": int(rt["steps_per_control"]),
      "onnx_fixed_batch_size": rt["fixed_batch_size"],
      "onnx_runtime_providers": list(rt["providers"]),
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


def _clip_eval_duration(clip: MotionClip, max_duration_s: float | None) -> float:
  duration_s = clip.length_s
  if max_duration_s is not None:
    duration_s = min(duration_s, max(float(max_duration_s), 0.0))
  return max(duration_s, 0.0)


def _onnx_fixed_batch_size(session: Any, input_name: str) -> int | None:
  del input_name
  batch_dim = session.get_inputs()[0].shape[0]
  return int(batch_dim) if isinstance(batch_dim, int) and batch_dim > 0 else None


def _run_onnx_actions(
  *,
  session: Any,
  input_name: str,
  output_name: str,
  obs_batch: np.ndarray,
  action_dim: int,
  fixed_batch_size: int | None,
) -> np.ndarray:
  if fixed_batch_size is None or fixed_batch_size == obs_batch.shape[0]:
    actions = session.run([output_name], {input_name: obs_batch})[0].astype(np.float32)
  elif fixed_batch_size == 1:
    actions = np.concatenate(
      [
        session.run([output_name], {input_name: obs_batch[idx : idx + 1]})[0]
        for idx in range(obs_batch.shape[0])
      ],
      axis=0,
    ).astype(np.float32)
  else:
    raise ValueError(
      "The YAHMP ONNX model has a fixed batch size "
      f"of {fixed_batch_size}, but the active batch has size {obs_batch.shape[0]}."
    )

  if actions.shape != (obs_batch.shape[0], action_dim):
    raise ValueError(
      f"Expected batched YAHMP actions [N,{action_dim}], got {actions.shape}."
    )
  return actions


def _build_runtime(cfg: EvaluateYahmpOnnxSuccessParallelConfig) -> dict[str, Any]:
  onnx_path = Path(cfg.onnx_path).expanduser().resolve()
  spec = PolicySpec.from_onnx(onnx_path)
  spec.validate()
  session, providers = _create_onnx_session(onnx_path, cfg.ort_provider)
  input_name = session.get_inputs()[0].name
  output_name = session.get_outputs()[0].name
  fixed_batch_size = _onnx_fixed_batch_size(session, input_name)

  model, _, scene_physics_dt, scene_control_dt = _build_task_scene(cfg.task_id)
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

  joint_qpos_adr, joint_qvel_adr = _joint_addresses(model, spec.joint_names)
  root_qpos_adr, root_qvel_adr = _root_addresses(model, spec.root_body_name)
  action_actuator_ids = _actuator_ids(model, spec.action_target_names)
  action_target_joint_indices = np.asarray(
    [spec.joint_names.index(name) for name in spec.action_target_names],
    dtype=np.int32,
  )
  steps_per_control = int(round(spec.control_dt / spec.physics_dt))
  if steps_per_control <= 0:
    raise ValueError(
      f"Invalid YAHMP control/physics dt ratio: control_dt={spec.control_dt}, "
      f"physics_dt={spec.physics_dt}."
    )
  ground_fail_body_ids = resolve_body_ids(model, cfg.ground_fail_body_names)
  hand_push_body_ids = resolve_body_ids(model, cfg.hand_push_body_names)

  return {
    "onnx_path": onnx_path,
    "spec": spec,
    "session": session,
    "providers": providers,
    "input_name": input_name,
    "output_name": output_name,
    "fixed_batch_size": fixed_batch_size,
    "model": model,
    "joint_qpos_adr": joint_qpos_adr,
    "joint_qvel_adr": joint_qvel_adr,
    "root_qpos_adr": root_qpos_adr,
    "root_qvel_adr": root_qvel_adr,
    "action_actuator_ids": action_actuator_ids,
    "action_target_joint_indices": action_target_joint_indices,
    "steps_per_control": steps_per_control,
    "ground_fail_body_ids": ground_fail_body_ids,
    "hand_push_body_ids": hand_push_body_ids,
  }


def run(cfg: EvaluateYahmpOnnxSuccessParallelConfig) -> dict[str, Any]:
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

  rt = _build_runtime(cfg)
  spec: PolicySpec = rt["spec"]
  model: mujoco.MjModel = rt["model"]
  data_pool = [mujoco.MjData(model) for _ in range(max(int(cfg.num_envs), 1))]

  for chunk_start in range(0, len(selected_indices), len(data_pool)):
    chunk = selected_indices[chunk_start : chunk_start + len(data_pool)]
    clips = [
      MotionClip(motion_files[motion_idx], root_body_name=spec.root_body_name)
      for motion_idx in chunk
    ]
    eval_duration_s = np.asarray(
      [_clip_eval_duration(clip, cfg.max_motion_duration_s) for clip in clips],
      dtype=np.float64,
    )
    total_steps = np.asarray(
      [
        int(math.ceil(duration_s / spec.control_dt)) if duration_s > 0.0 else 0
        for duration_s in eval_duration_s
      ],
      dtype=np.int64,
    )
    max_steps = int(total_steps.max()) if total_steps.size > 0 else 0

    histories: list[dict[str, np.ndarray]] = []
    previous_actions: list[np.ndarray] = []
    reference_contact_start = np.zeros(len(clips), dtype=bool)
    failed = np.zeros(len(clips), dtype=bool)
    completed_steps = np.zeros(len(clips), dtype=np.int64)
    metric_sums = {
      metric_name: np.zeros(len(clips), dtype=np.float64)
      for metric_name in TRACKING_METRIC_NAMES
    }
    metric_counts = np.zeros(len(clips), dtype=np.int64)
    actuation_stats = init_actuation_stats(len(clips), tuple(spec.action_target_names))
    hand_push_mirrors: list[HandPushMirror] = []

    for local_idx, clip in enumerate(clips):
      data = data_pool[local_idx]
      frame0 = clip.sample(0.0)
      _initialize_state(
        data,
        frame0,
        spec,
        rt["joint_qpos_adr"],
        rt["joint_qvel_adr"],
        rt["root_qpos_adr"],
        rt["root_qvel_adr"],
        init_default_joints=cfg.init_default_joints,
      )
      mujoco.mj_forward(model, data)
      reference_contact_start[local_idx] = ground_contact_active(
        model, data, rt["ground_fail_body_ids"]
      )
      previous_action = np.zeros(spec.action_dim, dtype=np.float32)
      terms = _term_values(
        model,
        data,
        spec,
        clip,
        0.0,
        frame0,
        rt["joint_qpos_adr"],
        rt["joint_qvel_adr"],
        previous_action,
      )
      histories.append(_initialize_history(spec, terms))
      previous_actions.append(previous_action)
      hand_push_mirrors.append(
        HandPushMirror(
          model=model,
          body_ids=rt["hand_push_body_ids"],
          dt=spec.control_dt,
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
      frame_by_idx: dict[int, Any] = {}
      for local_idx in active_indices:
        clip = clips[local_idx]
        data = data_pool[local_idx]
        time_s = min(float(step_idx) * spec.control_dt, eval_duration_s[local_idx])
        frame = clip.sample(safe_motion_time(clip, time_s))
        terms = _term_values(
          model,
          data,
          spec,
          clip,
          time_s,
          frame,
          rt["joint_qpos_adr"],
          rt["joint_qvel_adr"],
          previous_actions[local_idx],
        )
        obs = _build_observation(spec, terms, histories[local_idx])
        obs_batch.append(obs)
        frame_by_idx[int(local_idx)] = frame

      raw_actions = _run_onnx_actions(
        session=rt["session"],
        input_name=rt["input_name"],
        output_name=rt["output_name"],
        obs_batch=np.asarray(obs_batch, dtype=np.float32),
        action_dim=spec.action_dim,
        fixed_batch_size=rt["fixed_batch_size"],
      )

      for batch_idx, local_idx in enumerate(active_indices):
        data = data_pool[local_idx]
        raw_action = raw_actions[batch_idx]
        _apply_action(
          data,
          spec,
          raw_action,
          frame_by_idx[int(local_idx)],
          rt["action_actuator_ids"],
          rt["action_target_joint_indices"],
        )
        hand_push_mirrors[local_idx].pre_step(
          data,
          sample_clip_body_positions_w(
            clips[local_idx],
            min(float(step_idx) * spec.control_dt, eval_duration_s[local_idx]),
            cfg.hand_push_body_names,
          ),
        )
        for _ in range(rt["steps_per_control"]):
          mujoco.mj_step(model, data)
          update_actuation_stats(
            actuation_stats,
            int(local_idx),
            np.asarray(
              data.actuator_force[rt["action_actuator_ids"]], dtype=np.float64
            ),
          )
        hand_push_mirrors[local_idx].post_step(
          data,
          sample_clip_body_positions_w(
            clips[local_idx],
            min(float(step_idx + 1) * spec.control_dt, eval_duration_s[local_idx]),
            cfg.hand_push_body_names,
          ),
        )
        previous_actions[local_idx] = raw_action.copy()

      failed_by_eval = np.asarray(
        [
          ground_contact_active(model, data_pool[local_idx], rt["ground_fail_body_ids"])
          for local_idx in active_indices
        ],
        dtype=bool,
      )
      newly_failed_indices = active_indices[failed_by_eval]
      if newly_failed_indices.size > 0:
        failed[newly_failed_indices] = True

      survived_indices = active_indices[~failed_by_eval]
      for local_idx in survived_indices:
        clip = clips[local_idx]
        data = data_pool[local_idx]
        metric_time_s = min(
          float(step_idx + 1) * spec.control_dt, eval_duration_s[local_idx]
        )
        next_frame = clip.sample(safe_motion_time(clip, metric_time_s))
        next_terms = _term_values(
          model,
          data,
          spec,
          clip,
          metric_time_s,
          next_frame,
          rt["joint_qpos_adr"],
          rt["joint_qvel_adr"],
          previous_actions[local_idx],
        )
        _append_history(spec, histories[local_idx], next_terms)

        metrics = compute_tracking_metrics(
          model=model,
          data=data,
          clip=clip,
          time_s=metric_time_s,
          joint_qpos_adr=rt["joint_qpos_adr"],
          joint_qvel_adr=rt["joint_qvel_adr"],
          root_body_name=spec.root_body_name,
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
        "eval_duration_s": float(eval_duration_s[local_idx]),
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
    spec=spec,
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
    EvaluateYahmpOnnxSuccessParallelConfig,
    config=mjlab.TYRO_FLAGS,
  )
  result = run(cfg)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
