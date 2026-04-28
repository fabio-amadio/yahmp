"""YAHMP direct-PPO task configuration with future-motion encoding."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from yahmp import mdp
from yahmp.mdp import FutureJointRefAnchorRpMotionCommandCfg
from yahmp.yahmp_env_cfg import (
  _yahmp_history_term,
  _yahmp_privileged_terms,
  _yahmp_proprio_critic_terms,
  _yahmp_proprio_policy_terms,
)
from yahmp.yahmp_teacher_env_cfg import make_yahmp_teacher_env_cfg

DEFAULT_YAHMP_FUTURE_ENCODER_STEPS = (
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
  cfg = make_yahmp_teacher_env_cfg()
  base_motion_cmd = cfg.commands["motion"]

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
    entity_name=base_motion_cmd.entity_name,
    resampling_time_range=base_motion_cmd.resampling_time_range,
    debug_vis=base_motion_cmd.debug_vis,
    pose_range=base_motion_cmd.pose_range,
    velocity_range=base_motion_cmd.velocity_range,
    joint_position_range=base_motion_cmd.joint_position_range,
    motion_file=base_motion_cmd.motion_file,
    anchor_body_name=base_motion_cmd.anchor_body_name,
    body_names=base_motion_cmd.body_names,
    root_body_name=base_motion_cmd.root_body_name,
    command_step_offsets=(0, *DEFAULT_YAHMP_FUTURE_ENCODER_STEPS),
    sampling_mode=base_motion_cmd.sampling_mode,
  )

  return cfg
