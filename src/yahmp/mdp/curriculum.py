from __future__ import annotations

from typing import TYPE_CHECKING, TypedDict

import torch

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class FeasibleForceFractionRangeStage(TypedDict):
  step: int
  feasible_force_fraction_range: tuple[float, float]


def reset_reason_diagnostics(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
) -> dict[str, float]:
  """Snapshot reset-cause diagnostics for logging.

  The returned values are fractions over the environments being reset at this
  instant, with a simple precedence so causes remain mutually exclusive:

  1. failure
  2. time_out or motion_ref_expired
  """
  if env_ids is None:
    env_ids = slice(None)

  terminated = env.termination_manager.terminated[env_ids]
  motion_ref_expired = torch.zeros_like(terminated)
  time_out = torch.zeros_like(terminated)

  active_terms = env.termination_manager.active_terms
  if "motion_ref_expired" in active_terms:
    motion_ref_expired = env.termination_manager.get_term("motion_ref_expired")[env_ids]
  if "time_out" in active_terms:
    time_out = env.termination_manager.get_term("time_out")[env_ids]

  failure = terminated
  timeout = (~failure) & (motion_ref_expired | time_out)
  unclassified = ~(failure | timeout)

  return {
    "reset_count": float(failure.numel()),
    "failure_frac": failure.float().mean().item(),
    "time_out_frac": timeout.float().mean().item(),
    "unclassified_frac": unclassified.float().mean().item(),
  }


def event_feasible_force_fraction_range(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  event_name: str,
  feasible_force_fraction_stages: list[FeasibleForceFractionRangeStage],
) -> dict[str, torch.Tensor]:
  """Update a class-based event's ``feasible_force_fraction_range`` by step stages.

  This follows the same reset-time, step-threshold convention used by built-in
  ``mjlab`` curricula. The event config keeps the initial pre-threshold default,
  and later stages take over once ``env.common_step_counter`` crosses them.
  """
  del env_ids  # Unused.

  try:
    event_term_cfg = env.event_manager.get_term_cfg(event_name)
  except ValueError:
    return {}
  feasible_force_fraction_range = tuple(
    event_term_cfg.params["feasible_force_fraction_range"]
  )
  for stage in feasible_force_fraction_stages:
    if env.common_step_counter > stage["step"]:
      feasible_force_fraction_range = stage["feasible_force_fraction_range"]

  setter = getattr(event_term_cfg.func, "set_feasible_force_fraction_range", None)
  if setter is None or not callable(setter):
    raise TypeError(
      "Event term "
      f"'{event_name}' does not expose a callable set_feasible_force_fraction_range()."
    )
  setter(*feasible_force_fraction_range)
  low, high = feasible_force_fraction_range
  return {
    "low": torch.tensor(low),
    "high": torch.tensor(high),
  }


def event_force_magnitude_stats(
  env: ManagerBasedRlEnv,
  env_ids: torch.Tensor | slice | None,
  event_name: str,
) -> dict[str, torch.Tensor]:
  """Read reset-time force magnitude statistics from a class-based event term."""
  try:
    event_term_cfg = env.event_manager.get_term_cfg(event_name)
  except ValueError:
    return {}
  getter = getattr(event_term_cfg.func, "get_episode_force_magnitude_stats", None)
  if getter is None or not callable(getter):
    raise TypeError(
      "Event term "
      f"'{event_name}' does not expose a callable get_episode_force_magnitude_stats()."
    )
  return getter(env_ids=env_ids)
