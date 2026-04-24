"""YAHMP direct-PPO task configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise

from yahmp import mdp
from yahmp.mdp import JointRefAnchorRpMotionCommandCfg
from yahmp.yahmp_teacher_env_cfg import make_yahmp_teacher_env_cfg

YAHMP_HISTORY_LENGTH = 10


def _yahmp_proprio_policy_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
      noise=Unoise(n_min=-0.2, n_max=0.2),
    ),
    "projected_gravity": ObservationTermCfg(
      func=mdp.projected_gravity,
      noise=Unoise(n_min=-0.05, n_max=0.05),
    ),
    "joint_pos": ObservationTermCfg(
      func=mdp.joint_pos_rel,
      noise=Unoise(n_min=-0.01, n_max=0.01),
    ),
    "joint_vel": ObservationTermCfg(
      func=mdp.joint_vel_rel,
      noise=Unoise(n_min=-1.5, n_max=1.5),
    ),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def _yahmp_proprio_critic_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_ang_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_ang_vel"},
    ),
    "projected_gravity": ObservationTermCfg(func=mdp.projected_gravity),
    "joint_pos": ObservationTermCfg(func=mdp.joint_pos_rel),
    "joint_vel": ObservationTermCfg(func=mdp.joint_vel_rel),
    "actions": ObservationTermCfg(func=mdp.last_action),
  }


def _yahmp_privileged_terms() -> dict[str, ObservationTermCfg]:
  return {
    "base_lin_vel": ObservationTermCfg(
      func=mdp.builtin_sensor,
      params={"sensor_name": "robot/imu_lin_vel"},
    ),
    "motion_anchor_pos_b": ObservationTermCfg(
      func=mdp.motion_anchor_pos_b,
      params={"command_name": "motion"},
    ),
    "motion_anchor_ori_b": ObservationTermCfg(
      func=mdp.motion_anchor_ori_b,
      params={"command_name": "motion"},
    ),
    "body_pos": ObservationTermCfg(
      func=mdp.robot_body_pos_b,
      params={"command_name": "motion"},
    ),
    "body_ori": ObservationTermCfg(
      func=mdp.robot_body_ori_b,
      params={"command_name": "motion"},
    ),
    "feet_contact_mask": ObservationTermCfg(
      func=mdp.feet_contact_mask,
      params={"sensor_name": "feet_ground_contact"},
    ),
    "friction_coeff": ObservationTermCfg(
      func=mdp.motion_friction_coeff,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
      },
    ),
  }


def _yahmp_current_motion_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.motion_first_step_command,
    params={"command_name": "motion"},
  )


def _yahmp_history_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.YahmpObservationHistory,
    params={
      "command_name": "motion",
      "history_length": YAHMP_HISTORY_LENGTH,
    },
  )


def make_yahmp_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the YAHMP direct-PPO task template with history encoding only."""
  cfg = make_yahmp_teacher_env_cfg()
  base_motion_cmd = cfg.commands["motion"]

  policy_current_terms = _yahmp_proprio_policy_terms()
  actor_terms = {
    "command": _yahmp_current_motion_term(),
    **policy_current_terms,
    "history": _yahmp_history_term(),
  }

  critic_terms = {
    "command": _yahmp_current_motion_term(),
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

  cfg.commands["motion"] = JointRefAnchorRpMotionCommandCfg(
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
    sampling_mode=base_motion_cmd.sampling_mode,
  )

  return cfg
