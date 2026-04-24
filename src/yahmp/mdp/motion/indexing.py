"""Body-name resolution and indexing helpers for YAHMP motion sources."""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING

if TYPE_CHECKING:
  from .base import MotionCommand


def resolve_motion_body_names(command: MotionCommand) -> tuple[str, ...]:
  """Resolve body names for the motion tensors.

  Priority:
  1) Names embedded in the motion file (`body_names`).
  2) Names explicitly provided in the config (`motion_body_names`).
  3) Fallback to robot body names if tensor count matches exactly.
  """
  if command._uses_motion_library:
    assert command.motion_lib is not None
    assert command.motion_lib.body_names is not None
    motion_tensor_body_count = len(command.motion_lib.body_names)
    file_motion_body_names = command.motion_lib.body_names
  else:
    assert command.motion is not None
    motion_tensor_body_count = int(command.motion.body_pos_w.shape[1])
    file_motion_body_names = command.motion.body_names
  cfg_motion_body_names = command.cfg.motion_body_names

  if file_motion_body_names is not None:
    if len(file_motion_body_names) != motion_tensor_body_count:
      raise ValueError(
        "Motion file body name count does not match body tensor shape: "
        f"names={len(file_motion_body_names)} tensor_bodies={motion_tensor_body_count}"
      )
    if cfg_motion_body_names is not None and tuple(cfg_motion_body_names) != tuple(
      file_motion_body_names
    ):
      raise ValueError(
        "Both motion file and cfg.motion_body_names are provided but differ. "
        "Please keep only one source of truth or make them identical."
      )
    return tuple(file_motion_body_names)

  if cfg_motion_body_names is not None:
    if len(cfg_motion_body_names) != motion_tensor_body_count:
      raise ValueError(
        "cfg.motion_body_names count does not match motion body tensor shape: "
        f"names={len(cfg_motion_body_names)} tensor_bodies={motion_tensor_body_count}"
      )
    return tuple(cfg_motion_body_names)

  robot_body_names = tuple(command.robot.body_names)
  if motion_tensor_body_count == len(robot_body_names):
    warnings.warn(
      "Motion file has no body names and cfg.motion_body_names is unset. "
      "Falling back to robot body names because counts match.",
      stacklevel=2,
    )
    return robot_body_names

  raise ValueError(
    "Unable to resolve motion body names. Provide names in the motion source "
    "or set cfg.motion_body_names."
  )


def build_name_to_index(body_names: tuple[str, ...], source: str) -> dict[str, int]:
  """Build a unique body-name to index mapping and validate duplicates."""
  name_to_index: dict[str, int] = {}
  duplicates: list[str] = []
  for index, name in enumerate(body_names):
    if name in name_to_index:
      duplicates.append(name)
    else:
      name_to_index[name] = index
  if duplicates:
    raise ValueError(
      f"Duplicate body names found in {source} definition: {sorted(set(duplicates))}"
    )
  return name_to_index
