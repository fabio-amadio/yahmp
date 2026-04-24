from __future__ import annotations

import re
from pathlib import Path


def get_wandb_checkpoint_path(
  log_path: Path,
  run_path: Path,
  checkpoint_name: str | None = None,
) -> tuple[Path, bool]:
  """Get a checkpoint path from W&B, supporting both numbered and rolling-latest files.

  Returns:
    Tuple of `(checkpoint_path, was_cached)`.
  """
  import wandb

  run_id = str(run_path).split("/")[-1]
  download_dir = log_path / "wandb_checkpoints" / run_id

  api = wandb.Api()
  wandb_run = api.run(str(run_path))
  files = [file.name for file in wandb_run.files()]
  numbered_files = [
    file_name for file_name in files if re.match(r"^model_\d+\.pt$", file_name)
  ]
  has_latest = "model_latest.pt" in files

  if checkpoint_name is None:
    if numbered_files:
      checkpoint_file = max(
        numbered_files,
        key=lambda name: int(name.split("_")[1].split(".")[0]),
      )
    elif has_latest:
      checkpoint_file = "model_latest.pt"
    else:
      raise ValueError(
        f"No checkpoint found in run {run_path}. "
        "Expected numbered checkpoints like `model_1000.pt` or "
        "`model_latest.pt`."
      )
  else:
    if checkpoint_name not in files:
      available = sorted(
        file_name
        for file_name in files
        if file_name.endswith(".pt") or file_name.endswith(".onnx")
      )
      raise ValueError(
        f"Checkpoint '{checkpoint_name}' not found in run {run_path}. "
        f"Available: {available}"
      )
    checkpoint_file = checkpoint_name

  checkpoint_path = download_dir / checkpoint_file
  was_cached = checkpoint_path.exists()
  if not was_cached:
    download_dir.mkdir(parents=True, exist_ok=True)
    wandb_file = wandb_run.file(str(checkpoint_file))
    wandb_file.download(str(download_dir), replace=True)

  return checkpoint_path, was_cached
