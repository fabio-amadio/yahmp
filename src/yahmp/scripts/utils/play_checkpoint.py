"""Play a checkpoint from a W&B run."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import mjlab
import tyro
from mjlab.scripts.play import PlayConfig, run_play
from mjlab.tasks.registry import list_tasks, load_rl_cfg

from yahmp.utils import get_wandb_checkpoint_path


@dataclass(frozen=True)
class Config:
  wandb_run_path: str = ""
  agent: Literal["zero", "random", "trained"] = "trained"
  wandb_checkpoint_name: str | None = None
  checkpoint_file: str | None = None
  registry_name: str | None = None
  motion_file: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  no_terminations: bool = False


def main() -> None:
  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
    config=mjlab.TYRO_FLAGS,
  )

  cfg = tyro.cli(
    Config,
    args=remaining_args,
    default=Config(wandb_run_path=""),
    prog=sys.argv[0] + f" {chosen_task}",
    config=mjlab.TYRO_FLAGS,
  )

  wandb_run_path = cfg.wandb_run_path.strip() or None
  checkpoint_file = cfg.checkpoint_file.strip() if cfg.checkpoint_file else None

  if cfg.agent not in {"zero", "random"}:
    if checkpoint_file is not None:
      checkpoint_path = Path(checkpoint_file).expanduser().resolve()
      if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")
      print(f"[INFO]: Using local checkpoint: {checkpoint_path}")
    elif wandb_run_path is None:
      raise ValueError(
        "Using a trained agent requires either `--checkpoint-file` or "
        "`--wandb-run-path`. Alternatively use --agent zero/random."
      )
    else:
      agent_cfg = load_rl_cfg(chosen_task)
      log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
      checkpoint_path, was_cached = get_wandb_checkpoint_path(
        log_root_path,
        Path(wandb_run_path),
        cfg.wandb_checkpoint_name,
      )
      run_id = checkpoint_path.parent.name
      checkpoint_name = checkpoint_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Resolved checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
  else:
    checkpoint_path = None

  play_cfg = PlayConfig(
    agent=cfg.agent,
    registry_name=cfg.registry_name,
    wandb_run_path=wandb_run_path,
    wandb_checkpoint_name=None,
    checkpoint_file=str(checkpoint_path) if checkpoint_path is not None else None,
    motion_file=cfg.motion_file,
    num_envs=cfg.num_envs,
    device=cfg.device,
    video=cfg.video,
    video_length=cfg.video_length,
    video_height=cfg.video_height,
    video_width=cfg.video_width,
    camera=cfg.camera,
    viewer=cfg.viewer,
    no_terminations=cfg.no_terminations,
  )
  run_play(chosen_task, play_cfg)


if __name__ == "__main__":
  main()
