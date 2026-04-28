"""YAHMP teacher task configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg

from yahmp.yahmp_env_cfg import _history_term, _privileged_terms
from yahmp.yahmp_future_env_cfg import make_future_env_cfg


def make_teacher_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create YAHMP teacher task configuration template."""
  cfg = make_future_env_cfg()

  actor_group = cfg.observations["actor"]
  cfg.observations["actor"] = ObservationGroupCfg(
    terms={
      **actor_group.terms,
      **_privileged_terms(),
      "history": _history_term(include_privileged=True),
    },
    concatenate_terms=actor_group.concatenate_terms,
    enable_corruption=actor_group.enable_corruption,
  )

  return cfg
