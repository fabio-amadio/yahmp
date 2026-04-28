"""YAHMP base task configuration."""

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.tracking import mdp as tracking_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from yahmp import mdp
from yahmp.mdp import JointRefAnchorRpMotionCommandCfg

YAHMP_HISTORY_LENGTH = 10

PUSH_VELOCITY_RANGE = {
  "x": (-0.5, 0.5),
  "y": (-0.5, 0.5),
  "z": (-0.2, 0.2),
  "roll": (-0.52, 0.52),
  "pitch": (-0.52, 0.52),
  "yaw": (-0.78, 0.78),
}


def _yahmp_motion_command_kwargs() -> dict[str, object]:
  return {
    "entity_name": "robot",
    "resampling_time_range": (1.0e9, 1.0e9),
    "debug_vis": True,
    "pose_range": {
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    "velocity_range": PUSH_VELOCITY_RANGE,
    "joint_position_range": (-0.1, 0.1),
    # Set in robot cfg.
    "motion_file": "",
    "anchor_body_name": "",
    "body_names": (),
    "root_body_name": "",
    "sampling_mode": "adaptive",
  }


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
      params={"asset_cfg": SceneEntityCfg("robot", geom_names=())},
    ),
  }


def _yahmp_current_motion_term() -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.motion_first_step_command,
    params={"command_name": "motion"},
  )


def _yahmp_history_term(*, include_privileged: bool = False) -> ObservationTermCfg:
  return ObservationTermCfg(
    func=mdp.YahmpObservationHistory,
    params={
      "command_name": "motion",
      "history_length": YAHMP_HISTORY_LENGTH,
      "include_privileged": include_privileged,
    },
  )


def _yahmp_actions() -> dict[str, ActionTermCfg]:
  return {
    "joint_pos": mdp.ResidualJointPositionActionCfg(
      entity_name="robot",
      actuator_names=(".*",),
      scale=0.5,
      use_default_offset=False,
      command_name="motion",
    )
  }


def _yahmp_events() -> dict[str, EventTermCfg]:
  return {
    "push_robot": EventTermCfg(
      func=mdp.push_by_setting_velocity,
      mode="interval",
      interval_range_s=(1.0, 3.0),
      params={"velocity_range": PUSH_VELOCITY_RANGE},
    ),
    "push_end_effector": EventTermCfg(
      func=mdp.apply_torque_limited_body_force,
      mode="step",
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "duration_s": (0.5, 2.0),
        "cooldown_s": (0.0, 0.5),
        "joint_names": (
          "waist_.*_joint",
          ".*_shoulder_.*_joint",
          ".*_elbow_joint",
          ".*_wrist_.*_joint",
        ),
        "feasible_force_fraction_range": (0.05, 0.25),
        "max_force_magnitude": 20.0,
        "force_ramp_time_fraction": 0.15,
        "dirichlet_alpha": 1.0,
        "subtract_commanded_torque_margin": True,
        "use_current_qvel_for_inverse_dynamics": True,
        "body_point_offset": None,
        "randomize_application_point": False,
        "application_point_delta_range": None,
        "randomize_body": True,
        "eps": 1.0e-6,
        "debug_force_vis_enabled": True,
        "debug_force_vis_scale": 0.015,
        "debug_force_vis_width": 0.01,
      },
    ),
    # "base_mass": EventTermCfg(
    #   mode="startup",
    #   func=dr.pseudo_inertia,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot", body_names=()),  # Set in robot cfg.
    #     "alpha_range": (0.0, 0.0),  # Set per robot cfg.
    #   },
    # ),
    "base_com": EventTermCfg(
      mode="startup",
      func=dr.body_com_offset,
      params={
        "asset_cfg": SceneEntityCfg("robot", body_names=()),
        "operation": "add",
        "ranges": {
          0: (-0.025, 0.025),
          1: (-0.05, 0.05),
          2: (-0.05, 0.05),
        },
      },
    ),
    # "motor_strength": EventTermCfg(
    #   mode="startup",
    #   func=dr.pd_gains,
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "kp_range": (0.95, 1.05),
    #     "kd_range": (0.95, 1.05),
    #     "distribution": "uniform",
    #     "operation": "scale",
    #   },
    # ),
    "foot_friction": EventTermCfg(
      mode="startup",
      func=dr.geom_friction,
      params={
        "asset_cfg": SceneEntityCfg("robot", geom_names=()),
        "operation": "abs",
        "ranges": (0.3, 1.2),
        "shared_random": True,
      },
    ),
    # "action_delay": EventTermCfg(
    #   mode="interval",
    #   func=dr.sync_actuator_delays,
    #   interval_range_s=(0.02, 0.02),
    #   params={
    #     "asset_cfg": SceneEntityCfg("robot"),
    #     "lag_range": (0, 1),
    #   },
    # ),
  }


def _yahmp_rewards() -> dict[str, RewardTermCfg]:
  return {
    "motion_global_root_pos": RewardTermCfg(
      func=tracking_mdp.motion_global_anchor_position_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_global_root_ori": RewardTermCfg(
      func=tracking_mdp.motion_global_anchor_orientation_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_pos": RewardTermCfg(
      func=tracking_mdp.motion_relative_body_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_body_ori": RewardTermCfg(
      func=tracking_mdp.motion_relative_body_orientation_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.4},
    ),
    "motion_body_lin_vel": RewardTermCfg(
      func=tracking_mdp.motion_global_body_linear_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 1.0},
    ),
    "motion_body_ang_vel": RewardTermCfg(
      func=tracking_mdp.motion_global_body_angular_velocity_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 3.14},
    ),
    "motion_joint_pos": RewardTermCfg(
      func=mdp.motion_joint_position_error_exp,
      weight=1.0,
      params={"command_name": "motion", "std": 0.3},
    ),
    "motion_joint_vel": RewardTermCfg(
      func=mdp.motion_joint_velocity_error_exp,
      weight=0.5,
      params={"command_name": "motion", "std": 2.0},
    ),
    "feet_contact_forces": RewardTermCfg(
      func=mdp.feet_contact_force_excess,
      weight=-5.0e-4,
      params={"sensor_name": "feet_ground_contact", "max_contact_force": 300.0},
    ),
    "feet_slip": RewardTermCfg(
      func=mdp.feet_slip,
      weight=-0.1,
      params={
        "sensor_name": "feet_ground_contact",
        "asset_cfg": SceneEntityCfg(
          "robot",
          body_names=("left_ankle_roll_link", "right_ankle_roll_link"),
        ),
      },
    ),
    "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
    "joint_limit": RewardTermCfg(
      func=mdp.joint_pos_limits,
      weight=-10.0,
      params={"asset_cfg": SceneEntityCfg("robot", joint_names=(".*",))},
    ),
    # Keep this commented out while using the no-self-collision G1 preset.
    # "self_collisions": RewardTermCfg(
    #   func=tracking_mdp.self_collision_cost,
    #   weight=-10.0,
    #   params={"sensor_name": "self_collision"},
    # ),
  }


def _yahmp_terminations() -> dict[str, TerminationTermCfg]:
  return {
    "time_out": TerminationTermCfg(
      func=mdp.time_out,
      time_out=True,
    ),
    "motion_ref_expired": TerminationTermCfg(
      func=mdp.motion_ref_expired,
      time_out=True,
      params={"command_name": "motion"},
    ),
    "anchor_pos": TerminationTermCfg(
      func=tracking_mdp.bad_anchor_pos_z_only,
      params={"command_name": "motion", "threshold": 0.25},
    ),
    "anchor_ori": TerminationTermCfg(
      func=tracking_mdp.bad_anchor_ori,
      params={
        "asset_cfg": SceneEntityCfg("robot"),
        "command_name": "motion",
        "threshold": 0.8,
      },
    ),
    "ee_body_pos": TerminationTermCfg(
      func=tracking_mdp.bad_motion_body_pos_z_only,
      params={
        "command_name": "motion",
        "threshold": 0.25,
        "body_names": (),
      },
    ),
  }


def make_yahmp_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the YAHMP direct-PPO task template with history encoding only."""
  actor_terms = {
    "command": _yahmp_current_motion_term(),
    **_yahmp_proprio_policy_terms(),
    "history": _yahmp_history_term(),
  }

  critic_terms = {
    "command": _yahmp_current_motion_term(),
    **_yahmp_proprio_critic_terms(),
    "policy_history": _yahmp_history_term(),
    **_yahmp_privileged_terms(),
  }

  observations = {
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

  commands: dict[str, CommandTermCfg] = {
    "motion": JointRefAnchorRpMotionCommandCfg(**_yahmp_motion_command_kwargs())
  }

  return ManagerBasedRlEnvCfg(
    scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
    observations=observations,
    actions=_yahmp_actions(),
    commands=commands,
    events=_yahmp_events(),
    curriculum={},
    rewards=_yahmp_rewards(),
    terminations=_yahmp_terminations(),
    viewer=ViewerConfig(
      origin_type=ViewerConfig.OriginType.ASSET_BODY,
      entity_name="robot",
      body_name="",
      distance=3.0,
      elevation=-5.0,
      azimuth=90.0,
    ),
    sim=SimulationCfg(
      nconmax=50,
      njmax=400,
      mujoco=MujocoCfg(
        timestep=0.005,
        iterations=10,
        ls_iterations=20,
      ),
    ),
    decimation=4,
    episode_length_s=10.0,
  )
