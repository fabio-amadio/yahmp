import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from yahmp import mdp

HISTORY_LENGTH = 10
REACH_COMMAND_NAME = "reach"


PUSH_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def _reach_command_cfg() -> mdp.ReachTargetCommandCfg:
    return mdp.ReachTargetCommandCfg(
        entity_name="robot",
        resampling_time_range=(8.0, 12.0),
        debug_vis=True,
        left_wrist_body_name="left_wrist_yaw_link",
        right_wrist_body_name="right_wrist_yaw_link",
        p_right=0.5,
        distance_range=(0.0, 3.0),
        angle_range=(0.0, 0.2),
        height_range=(1.05, 1.30),
        reach_tol=0.12,
        standoff_radius=0.30,
        v_target_range=(0.6, 1.2),
        time_clip=3.0,
        arrive_radius=0.20,
        arrive_speed=0.30,
    )


def _proprio_actor_terms() -> dict[str, ObservationTermCfg]:
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


def _proprio_critic_terms() -> dict[str, ObservationTermCfg]:
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


def _privileged_terms() -> dict[str, ObservationTermCfg]:
    return {
        "base_lin_vel": ObservationTermCfg(
            func=mdp.builtin_sensor,
            params={"sensor_name": "robot/imu_lin_vel"},
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


def _command_term() -> ObservationTermCfg:
    return ObservationTermCfg(
        func=mdp.reach_command,
        params={"command_name": REACH_COMMAND_NAME},
    )


def _history_term(*, include_privileged: bool = False) -> ObservationTermCfg:
    return ObservationTermCfg(
        func=mdp.YahmpLocomotionObservationHistory,
        params={
            "command_name": REACH_COMMAND_NAME,
            "history_length": HISTORY_LENGTH,
            "include_privileged": include_privileged,
        },
    )


def _actions() -> dict[str, ActionTermCfg]:
    return {
        "joint_pos": JointPositionActionCfg(
            entity_name="robot",
            actuator_names=(".*",),
            scale=0.5,
            use_default_offset=True,
        )
    }


def _events() -> dict[str, EventTermCfg]:
    return {
        "reset_base": EventTermCfg(
            func=vel_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "pose_range": {
                    "x": (-0.5, 0.5),
                    "y": (-0.5, 0.5),
                    "z": (0.0, 0.0),
                    "yaw": (-math.pi, math.pi),
                },
                "velocity_range": {},
            },
        ),
        "reset_robot_joints": EventTermCfg(
            func=vel_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot", joint_names=(".*",)),
            },
        ),
        "push_robot": EventTermCfg(
            func=mdp.push_moving_envs_by_robot_speed,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={
                "velocity_range": PUSH_VELOCITY_RANGE,
                "speed_threshold": 0.3,
            },
        ),
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
        "foot_friction": EventTermCfg(
            mode="startup",
            func=dr.geom_friction,
            params={
                "asset_cfg": SceneEntityCfg("robot", geom_names=()),
                "operation": "abs",
                "ranges": (0.4, 1.2),
                "shared_random": True,
            },
        ),
    }


def _rewards() -> dict[str, mdp.RewardTermCfg]:
    from mjlab.managers.reward_manager import RewardTermCfg

    return {
        "reach_position": RewardTermCfg(
            func=mdp.reach_position_tracking_exp,
            weight=1.0,
            params={
                "command_name": REACH_COMMAND_NAME,
                "std": 0.5,
                "phase_gated": True,
            },
        ),
        "reach_position_fine": RewardTermCfg(
            func=mdp.reach_position_tracking_exp,
            weight=2.0,
            params={
                "command_name": REACH_COMMAND_NAME,
                "std": 0.15,
                "phase_gated": True,
            },
        ),
        "reach_success": RewardTermCfg(
            func=mdp.reach_success_bonus,
            weight=1.0,
            params={"command_name": REACH_COMMAND_NAME},
        ),
        "reach_pace": RewardTermCfg(
            func=mdp.reach_pace_exp,
            weight=2.0,
            params={"command_name": REACH_COMMAND_NAME, "std": 0.5},
        ),
        "reach_stop": RewardTermCfg(
            func=mdp.reach_stop_at_target,
            weight=1.0,
            params={"command_name": REACH_COMMAND_NAME, "speed_std": 0.3},
        ),
        "base_overspeed": RewardTermCfg(
            func=mdp.reach_base_overspeed_penalty,
            weight=-0.5,
            params={"command_name": REACH_COMMAND_NAME, "max_speed": 1.5},
        ),
        # Structural anti-body-bump: pelvis must keep its distance; the hand
        # covers the rest. Soft (metres of intrusion).
        "base_standoff": RewardTermCfg(
            func=mdp.reach_base_standoff_penalty,
            weight=-1.0,
            params={"command_name": REACH_COMMAND_NAME},
        ),
        "wrong_hand": RewardTermCfg(
            func=mdp.reach_wrong_hand_penalty,
            weight=-0.5,
            params={"command_name": REACH_COMMAND_NAME},
        ),
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-0.5,
            params={"sensor_name": "self_collision"},
        ),
    }


def _terminations() -> dict[str, TerminationTermCfg]:
    return {
        "time_out": TerminationTermCfg(func=mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=vel_mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        ),
    }


def _curriculum() -> dict[str, CurriculumTermCfg]:
    return {
        "reach_levels": CurriculumTermCfg(
            func=mdp.reach_command_levels,
            params={
                "command_name": REACH_COMMAND_NAME,
                "stages": [
                    {
                        "step": 0,
                        "distance_range": (0.0, 0.5),
                        "angle_range": (-0.5, 0.5),
                        "v_target_range": (0.8, 1.2),
                    },
                    {
                        "step": 800 * 24,
                        "distance_range": (0.0, 1.0),
                        "angle_range": (-1.0, 1.0),
                        "v_target_range": (0.7, 1.2),
                    },
                    {
                        "step": 1600 * 24,
                        "distance_range": (0.0, 2.0),
                        "angle_range": (-2.0, 2.0),
                        "v_target_range": (0.6, 1.2),
                    },
                    {
                        "step": 3000 * 24,
                        "distance_range": (0.0, 4.0),
                        "angle_range": (-math.pi, math.pi),
                        "v_target_range": (0.6, 1.2),
                    },
                ],
            },
        ),
    }


def make_locomanip_env_cfg() -> ManagerBasedRlEnvCfg:
    """YAHMP point-goal hand-reach task template (residual joint actions)."""
    actor_terms = {
        "command": _command_term(),
        **_proprio_actor_terms(),
        "history": _history_term(),
    }
    critic_terms = {
        "command": _command_term(),
        **_proprio_critic_terms(),
        "policy_history": _history_term(),
        **_privileged_terms(),
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
        REACH_COMMAND_NAME: _reach_command_cfg(),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
        observations=observations,
        actions=_actions(),
        commands=commands,
        events=_events(),
        curriculum=_curriculum(),
        rewards=_rewards(),
        terminations=_terminations(),
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
        episode_length_s=15.0,
    )
