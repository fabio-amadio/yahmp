"""Export a checkpoint to ONNX."""

from __future__ import annotations

import argparse
from dataclasses import asdict
from pathlib import Path

import mjlab.tasks  # noqa: F401
import torch
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg

from yahmp.rl.exporter import attach_onnx_metadata
from yahmp.rl.runner import _normalize_gaussian_distribution_state_dict
from yahmp.utils import get_wandb_checkpoint_path


def _load_runner_for_export(runner, checkpoint_path: Path, device: str) -> None:
  loaded_dict = torch.load(checkpoint_path, map_location=device, weights_only=False)
  runner_cfg = getattr(runner, "cfg", {})
  has_actor_model = isinstance(runner_cfg, dict) and "actor" in runner_cfg

  if "actor_state_dict" in loaded_dict and has_actor_model and hasattr(runner, "alg"):
    actor_state_dict = dict(loaded_dict["actor_state_dict"])
    _normalize_gaussian_distribution_state_dict(actor_state_dict)
    loaded_dict["actor_state_dict"] = actor_state_dict
    runner.alg.load(
      loaded_dict,
      load_cfg={
        "actor": True,
        "critic": False,
        "optimizer": False,
        "iteration": False,
        "rnd": False,
      },
      strict=True,
    )
    return

  runner.load(str(checkpoint_path), map_location=device)


def export_checkpoint_to_onnx(
  *,
  task_id: str,
  checkpoint_path: Path,
  output_path: Path,
  device: str,
  num_envs: int,
) -> None:
  env_cfg = load_env_cfg(task_id)
  agent_cfg = load_rl_cfg(task_id)

  if num_envs > 0:
    env_cfg.scene.num_envs = num_envs

  is_tracking_task = "motion" in env_cfg.commands and isinstance(
    env_cfg.commands["motion"], MotionCommandCfg
  )
  if is_tracking_task:
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg)
    if not motion_cmd.motion_file:
      raise ValueError(
        "Tracking task export needs a motion source, but this task config has no "
        "default `motion_file`."
      )

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  try:
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)
    runner_cls = load_runner_cls(task_id) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    _load_runner_for_export(runner, checkpoint_path, device)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    runner.export_policy_to_onnx(str(output_path.parent), output_path.name)
    attach_onnx_metadata(env.unwrapped, checkpoint_path.stem, str(output_path))
  finally:
    env.close()


def _build_argparser() -> argparse.ArgumentParser:
  parser = argparse.ArgumentParser(
    description="Export a checkpoint to ONNX with deployment metadata."
  )
  parser.add_argument(
    "--task-id",
    required=True,
    choices=sorted(list_tasks()),
    help="Registered mjlab task id.",
  )
  checkpoint_group = parser.add_mutually_exclusive_group(required=True)
  checkpoint_group.add_argument(
    "--checkpoint-file",
    type=Path,
    help="Checkpoint file to export.",
  )
  checkpoint_group.add_argument(
    "--wandb-run-path",
    type=str,
    default=None,
    help="W&B run path like entity/project/run_id.",
  )
  parser.add_argument(
    "--wandb-checkpoint-name",
    type=str,
    default=None,
    help="Optional checkpoint filename within the W&B run, e.g. model_7000.pt.",
  )
  parser.add_argument(
    "--output-path",
    type=Path,
    default=None,
    help="Target ONNX file. Defaults to <checkpoint>.onnx.",
  )
  parser.add_argument(
    "--device",
    default="cpu",
    help="Torch device used to instantiate and load the policy.",
  )
  parser.add_argument(
    "--num-envs",
    type=int,
    default=1,
    help="Number of envs to instantiate during export. Keep this small.",
  )
  return parser


def main() -> None:
  parser = _build_argparser()
  args = parser.parse_args()

  if args.checkpoint_file is not None:
    checkpoint_path = args.checkpoint_file.expanduser().resolve()
    if not checkpoint_path.is_file():
      parser.error(f"Checkpoint file not found: {checkpoint_path}")
  else:
    agent_cfg = load_rl_cfg(str(args.task_id))
    log_root_path = Path("logs") / "rsl_rl" / agent_cfg.experiment_name
    checkpoint_path, was_cached = get_wandb_checkpoint_path(
      log_root_path.resolve(),
      Path(str(args.wandb_run_path)),
      args.wandb_checkpoint_name,
    )
    cache_state = "cached" if was_cached else "downloaded"
    print(f"[INFO] Using W&B checkpoint: {checkpoint_path} ({cache_state})")

  output_path = (
    args.output_path.expanduser().resolve()
    if args.output_path is not None
    else checkpoint_path.with_suffix(".onnx")
  )

  if args.device.startswith("cuda") and not torch.cuda.is_available():
    parser.error(
      f"Requested device `{args.device}` but CUDA is not available in this environment."
    )

  export_checkpoint_to_onnx(
    task_id=str(args.task_id),
    checkpoint_path=checkpoint_path,
    output_path=output_path,
    device=str(args.device),
    num_envs=int(args.num_envs),
  )
  print(f"[INFO] Exported ONNX to: {output_path}")


if __name__ == "__main__":
  main()
