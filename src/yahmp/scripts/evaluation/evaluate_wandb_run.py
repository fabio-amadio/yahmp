"""Export and evaluate a YAHMP policy stored in a W&B run."""

from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import mjlab
import tyro
from mjlab.tasks.registry import load_rl_cfg

from yahmp.scripts.deploy.export_checkpoint_to_onnx import export_checkpoint_to_onnx
from yahmp.scripts.evaluation.evaluate_yahmp_onnx_success_parallel import (
  EvaluateYahmpOnnxSuccessParallelConfig,
)
from yahmp.scripts.evaluation.evaluate_yahmp_onnx_success_parallel import (
  run as run_onnx_evaluation,
)
from yahmp.scripts.evaluation.tracking_eval_utils import (
  resolve_motion_files,
  resolve_motion_source,
)
from yahmp.utils import get_wandb_checkpoint_path

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "assets/logs/hum2026_cross_eval"


@dataclass(frozen=True)
class EvaluateWandbRunConfig:
  """Configuration for W&B checkpoint export and held-out motion evaluation."""

  task_id: str
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  wandb_checkpoint_name: str | None = None
  motion_source: str | None = None
  output_dir: str | None = None
  export_device: str = "cpu"
  export_num_envs: int = 1
  force_export: bool = False
  ort_provider: str = "auto"
  num_envs: int = 256
  max_motions: int | None = None
  start_motion_index: int = 0
  max_motion_duration_s: float | None = None
  init_default_joints: bool = False
  skip_reference_ground_contact_start: bool = True
  enable_hand_pushes: bool = False
  hand_push_seed: int = 0
  resume: bool = False


def _slug(value: str) -> str:
  slug = re.sub(r"[^A-Za-z0-9._-]+", "-", value.strip()).strip("-._")
  return slug or "unnamed"


def _run_id(wandb_run_path: str) -> str:
  parts = [part for part in wandb_run_path.strip().strip("/").split("/") if part]
  if not parts:
    raise ValueError("`wandb_run_path` must not be empty.")
  return parts[-1]


def _default_output_dir(task_id: str, run_name: str) -> Path:
  task_slug = _slug(task_id.removeprefix("Mjlab-").removesuffix("-Unitree-G1"))
  return DEFAULT_OUTPUT_ROOT / task_slug / _slug(run_name)


def _write_metadata(path: Path, payload: dict[str, Any]) -> None:
  path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")


def _validate_task_id(task_id: str) -> None:
  from mjlab.tasks.registry import list_tasks

  available = set(list_tasks())
  if task_id not in available:
    yahmp_tasks = sorted(task for task in available if "YAHMP" in task)
    raise ValueError(
      f"Unknown task ID `{task_id}`. Available YAHMP tasks: {yahmp_tasks}"
    )


def run(cfg: EvaluateWandbRunConfig) -> dict[str, Any]:
  import mjlab.tasks  # noqa: F401

  import yahmp.config.g1  # noqa: F401

  _validate_task_id(cfg.task_id)
  agent_cfg = load_rl_cfg(cfg.task_id)
  log_root = (REPO_ROOT / "logs/rsl_rl" / agent_cfg.experiment_name).resolve()
  if cfg.checkpoint_file is not None:
    checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.is_file():
      raise FileNotFoundError(f"Local checkpoint not found: {checkpoint_path}")
    was_cached = False
    checkpoint_source = "local"
    run_id = checkpoint_path.parent.name
  else:
    if cfg.wandb_run_path is None:
      raise ValueError("Provide either `--checkpoint-file` or `--wandb-run-path`.")
    checkpoint_path, was_cached = get_wandb_checkpoint_path(
      log_root,
      Path(cfg.wandb_run_path),
      cfg.wandb_checkpoint_name,
    )
    checkpoint_source = "wandb"
    run_id = _run_id(cfg.wandb_run_path)

  output_dir = (
    Path(cfg.output_dir).expanduser().resolve()
    if cfg.output_dir is not None
    else _default_output_dir(cfg.task_id, run_id).resolve()
  )
  output_dir.mkdir(parents=True, exist_ok=True)
  onnx_path = output_dir / "policy.onnx"

  resolved_motion_source = resolve_motion_source(cfg.task_id, cfg.motion_source)
  motion_files = resolve_motion_files(resolved_motion_source)
  motion_folders = sorted({str(path.parent) for path in motion_files})
  if cfg.wandb_run_path is not None:
    print(f"[INFO] W&B run: {cfg.wandb_run_path}")
  print(f"[INFO] Task ID: {cfg.task_id}")
  print(f"[INFO] Checkpoint: {checkpoint_path}")
  print(f"[INFO] Motion source config: {resolved_motion_source}")
  print(f"[INFO] Motion source folder(s): {motion_folders}")
  print(f"[INFO] Output directory: {output_dir}")

  if cfg.force_export or not onnx_path.exists():
    print(f"[INFO] Exporting policy to ONNX: {onnx_path}")
    export_checkpoint_to_onnx(
      task_id=cfg.task_id,
      checkpoint_path=checkpoint_path,
      output_path=onnx_path,
      device=cfg.export_device,
      num_envs=cfg.export_num_envs,
    )
  else:
    print(f"[INFO] Reusing exported ONNX: {onnx_path}")

  evaluation_cfg = EvaluateYahmpOnnxSuccessParallelConfig(
    task_id=cfg.task_id,
    onnx_path=str(onnx_path),
    motion_source=resolved_motion_source,
    output_dir=str(output_dir),
    ort_provider=cfg.ort_provider,
    num_envs=cfg.num_envs,
    max_motions=cfg.max_motions,
    start_motion_index=cfg.start_motion_index,
    max_motion_duration_s=cfg.max_motion_duration_s,
    init_default_joints=cfg.init_default_joints,
    skip_reference_ground_contact_start=cfg.skip_reference_ground_contact_start,
    enable_hand_pushes=cfg.enable_hand_pushes,
    hand_push_seed=cfg.hand_push_seed,
    resume=cfg.resume,
  )

  metadata_path = output_dir / "wandb_evaluation_metadata.json"
  metadata: dict[str, Any] = {
    "wandb": {
      "run_path": cfg.wandb_run_path,
      "run_id": run_id,
      "checkpoint_name": checkpoint_path.name,
      "checkpoint_path": str(checkpoint_path.resolve()),
      "checkpoint_source": checkpoint_source,
      "checkpoint_was_cached": was_cached,
    },
    "policy": {
      "task_id": cfg.task_id,
      "onnx_path": str(onnx_path),
    },
    "motion_source": {
      "config_path": resolved_motion_source,
      "folders": motion_folders,
    },
    "driver_config": asdict(cfg),
    "evaluation_config": asdict(evaluation_cfg),
    "status": "running",
  }
  _write_metadata(metadata_path, metadata)

  try:
    result = run_onnx_evaluation(evaluation_cfg)
  except Exception:
    metadata["status"] = "failed"
    _write_metadata(metadata_path, metadata)
    raise

  metadata["status"] = "complete"
  metadata["outputs"] = {
    key: value for key, value in result.items() if key != "summary"
  }
  _write_metadata(metadata_path, metadata)
  result["wandb_evaluation_metadata_json"] = str(metadata_path)
  return result


def main() -> None:
  cfg = tyro.cli(EvaluateWandbRunConfig, config=mjlab.TYRO_FLAGS)
  result = run(cfg)
  print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
  main()
