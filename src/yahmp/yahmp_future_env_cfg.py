"""YAHMP direct-PPO task configuration with future-motion encoding."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from yahmp import mdp
from yahmp.mdp import FutureJointRefAnchorRpMotionCommandCfg
from yahmp.yahmp_env_cfg import (
  _history_term,
  _motion_command_kwargs,
  _privileged_terms,
  _proprio_critic_terms,
  _proprio_policy_terms,
  make_env_cfg,
)

FUTURE_STEPS = (
  4,
  8,
  12,
  16,
  20,
  24,
  28,
  32,
  36,
  40,
  44,
  48,
)


def _future_motion_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "motion"},
  )


def make_future_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the YAHMP task template with future-motion and history encoders."""
  cfg = make_env_cfg()

  actor_terms = {
    "command": _future_motion_term(),
    **_proprio_policy_terms(),
    "history": _history_term(),
  }

  critic_terms = {
    "command": _future_motion_term(),
    **_proprio_critic_terms(),
    "policy_history": _history_term(),
    **_privileged_terms(),
  }

  cfg.observations = {
    "actor": ObservationGroupCfg(
      terms=actor_terms,
      concatenate_terms=True,
      enable_corruption=True,
    ),
    "critic": ObservationGroupCfg(
      terms=critic_terms,
      concatenate_terms=True,
      enable_corruption=False,
    ),
  }

  cfg.commands["motion"] = FutureJointRefAnchorRpMotionCommandCfg(
    **_motion_command_kwargs(),
    command_step_offsets=(0, *FUTURE_STEPS),
  )

  return cfg
