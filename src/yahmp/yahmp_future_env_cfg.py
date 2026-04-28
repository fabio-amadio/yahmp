"""YAHMP direct-PPO task configuration with future-motion encoding."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from yahmp import mdp
from yahmp.mdp import FutureJointRefAnchorRpMotionCommandCfg
from yahmp.yahmp_env_cfg import (
  _yahmp_history_term,
  _yahmp_motion_command_kwargs,
  _yahmp_privileged_terms,
  _yahmp_proprio_critic_terms,
  _yahmp_proprio_policy_terms,
  make_yahmp_env_cfg,
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


def _yahmp_future_motion_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.generated_commands,
    params={"command_name": "motion"},
  )


def make_yahmp_future_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the YAHMP task template with future-motion and history encoders."""
  cfg = make_yahmp_env_cfg()

  actor_terms = {
    "command": _yahmp_future_motion_term(),
    **_yahmp_proprio_policy_terms(),
    "history": _yahmp_history_term(),
  }

  critic_terms = {
    "command": _yahmp_future_motion_term(),
    **_yahmp_proprio_critic_terms(),
    "policy_history": _yahmp_history_term(),
    **_yahmp_privileged_terms(),
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
    **_yahmp_motion_command_kwargs(),
    command_step_offsets=(0, *FUTURE_STEPS),
  )

  return cfg
