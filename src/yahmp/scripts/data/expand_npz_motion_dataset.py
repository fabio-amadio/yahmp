"""Expand compact YAHMP NPZ motion files into fully materialized NPZ files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from tqdm import tqdm

from yahmp.mdp.motion.library import load_motion_file


@dataclass(frozen=True)
class Args:
  input_root: Path
  output_root: Path
  overwrite: bool
  compressed: bool
  fail_on_error: bool


def _default_input_root() -> Path:
  return (
    Path(__file__).resolve().parents[4]
    / "assets"
    / "motions"
    / "g1_omomo_amass_clean"
  )


def _default_output_root(input_root: Path) -> Path:
  return input_root.parent / f"{input_root.name}_full"


def _parse_args() -> Args:
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
    "--input-root",
    type=Path,
    default=_default_input_root(),
    help="Root folder containing source NPZ files (subfolders are preserved).",
  )
  parser.add_argument(
    "--output-root",
    type=Path,
    default=None,
    help="Root folder where expanded NPZ files are written.",
  )
  parser.add_argument(
    "--overwrite",
    action="store_true",
    help="Overwrite existing NPZ files.",
  )
  parser.add_argument(
    "--uncompressed",
    action="store_true",
    help="Use np.savez (default uses np.savez_compressed).",
  )
  parser.add_argument(
    "--fail-on-error",
    action="store_true",
    help="Stop on first conversion error.",
  )
  ns = parser.parse_args()

  input_root = ns.input_root.expanduser().resolve()
  output_root = (
    _default_output_root(input_root)
    if ns.output_root is None
    else ns.output_root.expanduser().resolve()
  )
  return Args(
    input_root=input_root,
    output_root=output_root,
    overwrite=ns.overwrite,
    compressed=not ns.uncompressed,
    fail_on_error=ns.fail_on_error,
  )


def _load_original_arrays(npz_path: Path) -> dict[str, np.ndarray]:
  with np.load(npz_path, allow_pickle=False) as data:
    return {key: np.asarray(data[key]) for key in data.files}


def _expanded_payload(npz_path: Path) -> dict[str, np.ndarray]:
  original = _load_original_arrays(npz_path)
  motion = load_motion_file(npz_path)

  payload = dict(original)
  payload.update(
    {
      "fps": np.asarray([motion.fps], dtype=np.float64),
      "joint_pos": motion.joint_pos.astype(np.float32, copy=False),
      "joint_vel": motion.joint_vel.astype(np.float32, copy=False),
      "body_pos_w": motion.body_pos_w.astype(np.float32, copy=False),
      "body_quat_w": motion.body_quat_w.astype(np.float32, copy=False),
      "body_lin_vel_w": motion.body_lin_vel_w.astype(np.float32, copy=False),
      "body_ang_vel_w": motion.body_ang_vel_w.astype(np.float32, copy=False),
      "body_names": np.asarray(motion.body_names, dtype=str),
    }
  )
  return payload


def _convert_one(npz_path: Path, out_path: Path, args: Args) -> tuple[bool, str]:
  if out_path.exists() and not args.overwrite:
    return False, "skip: npz exists"

  payload = _expanded_payload(npz_path)
  out_path.parent.mkdir(parents=True, exist_ok=True)
  if args.compressed:
    np.savez_compressed(out_path, **payload)
  else:
    np.savez(out_path, **payload)
  return True, "ok"


def main() -> None:
  args = _parse_args()
  if not args.input_root.exists():
    raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
  if not args.input_root.is_dir():
    raise NotADirectoryError(f"Input root must be a directory: {args.input_root}")

  npz_files = sorted(args.input_root.rglob("*.npz"))
  if len(npz_files) == 0:
    print(f"No NPZ files found in: {args.input_root}")
    return

  converted = 0
  skipped = 0
  failed = 0
  progress = tqdm(npz_files, desc="Expanding NPZ motions", unit="file")
  for npz_path in progress:
    rel = npz_path.relative_to(args.input_root)
    out_path = args.output_root / rel
    try:
      changed, msg = _convert_one(npz_path=npz_path, out_path=out_path, args=args)
      if changed:
        converted += 1
      else:
        skipped += 1
      progress.set_postfix_str(
        f"converted={converted} skipped={skipped} failed={failed}"
      )
      if msg != "ok":
        tqdm.write(f"{npz_path}: {msg}")
    except Exception as exc:  # noqa: BLE001
      failed += 1
      tqdm.write(f"{npz_path}: failed: {exc}")
      if args.fail_on_error:
        raise

  print(
    "Done. "
    f"converted={converted} skipped={skipped} failed={failed} total={len(npz_files)}"
  )


if __name__ == "__main__":
  main()
