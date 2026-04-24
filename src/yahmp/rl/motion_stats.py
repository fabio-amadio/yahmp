"""Utilities for writing compact per-motion training diagnostics."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import wandb
from mjlab.rl import RslRlVecEnvWrapper

from yahmp.mdp.motion.base import MotionCommand


def dump_motion_stats(
  env: RslRlVecEnvWrapper,
  cfg: dict[str, Any],
  checkpoint_path: Path,
  iteration: int,
  *,
  logger_type: str | None = None,
) -> None:
  """Write compact per-motion stats with rolling-latest as the default mode."""
  mode = str(cfg.get("motion_stats_mode", "rolling_latest"))
  if mode in {"disabled", "none", "off", "false"}:
    return
  if mode not in {"rolling_latest", "all"}:
    raise ValueError(
      f"Unsupported motion_stats_mode `{mode}`. "
      "Expected one of: disabled, rolling_latest, all."
    )

  motion_term = env.unwrapped.command_manager.get_term("motion")
  if not isinstance(motion_term, MotionCommand):
    return

  stats_dir = checkpoint_path.parent / "motion_stats"
  latest_path = stats_dir / "motion_stats_latest.csv"
  if not motion_term.write_motion_stats_csv(latest_path):
    return

  upload = bool(cfg.get("upload_motion_stats", cfg.get("upload_model", True)))
  if logger_type == "wandb" and wandb.run is not None and upload:
    wandb.save(str(latest_path), base_path=str(stats_dir))

  if mode == "all":
    snapshot_path = stats_dir / f"motion_stats_{int(iteration):08d}.csv"
    motion_term.write_motion_stats_csv(snapshot_path)
    if logger_type == "wandb" and wandb.run is not None and upload:
      wandb.save(str(snapshot_path), base_path=str(stats_dir))
