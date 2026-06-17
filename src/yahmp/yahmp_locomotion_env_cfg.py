"""YAHMP omnidirectional locomotion task configuration.

Velocity-command tracking environment used to learn a high-level policy on top
of a frozen YAHMP imitation backbone (history_encoder is rebuilt; prior, RVQ
and action_decoder are loaded from the imitation checkpoint and frozen).

Goal: ``g_task = (vx_cmd, vy_cmd, ω_z_cmd)`` in body frame.
"""

import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp import dr
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.action_manager import ActionTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.managers.termination_manager import TerminationTermCfg
from mjlab.scene import SceneCfg
from mjlab.sim import MujocoCfg, SimulationCfg
from mjlab.tasks.velocity import mdp as vel_mdp
from mjlab.tasks.velocity.mdp import UniformVelocityCommandCfg
from mjlab.terrains import TerrainEntityCfg
from mjlab.utils.noise import UniformNoiseCfg as Unoise
from mjlab.viewer import ViewerConfig

from yahmp import mdp

HISTORY_LENGTH = 10
TWIST_COMMAND_NAME = "twist"

PUSH_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}


def _velocity_command_kwargs() -> dict[str, object]:
    return {
        "entity_name": "robot",
        "resampling_time_range": (5.0, 10.0),
        "rel_standing_envs": 0.1,
        "rel_heading_envs": 0.0,
        "heading_command": False,
        "debug_vis": True,
        "ranges": UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(-0.7, 1.5),
            lin_vel_y=(-0.7, 0.7),
            ang_vel_z=(-1.2, 1.2),
        ),
    }


def _proprio_actor_terms() -> dict[str, ObservationTermCfg]:
    """Same proprio block used by the YAHMP imitation actor, with corruption."""
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
        func=mdp.velocity_command,
        params={"command_name": TWIST_COMMAND_NAME},
    )


def _history_term(*, include_privileged: bool = False) -> ObservationTermCfg:
    return ObservationTermCfg(
        func=mdp.YahmpLocomotionObservationHistory,
        params={
            "command_name": TWIST_COMMAND_NAME,
            "history_length": HISTORY_LENGTH,
            "include_privileged": include_privileged,
        },
    )


def _actions() -> dict[str, ActionTermCfg]:
    """Residual joint position action relative to the default pose.

    Locomotion has no motion reference, so the residual baseline is the
    robot's default joint pose (``use_default_offset=True``). The frozen
    imitation ``action_decoder`` produces small offsets that we add on top
    of that baseline before sending to the PD controller.
    """
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
        # Only perturb envs that are commanded to move. Pushing standing-
        # commanded envs taught the policy reactive arm/torso balance at rest
        # (standing jitter); gating on command magnitude keeps the perturbation
        # robustness during walking/running while sparing standstill.
        "push_robot": EventTermCfg(
            func=mdp.push_moving_envs_by_setting_velocity,
            mode="interval",
            interval_range_s=(3.0, 6.0),
            params={
                "velocity_range": PUSH_VELOCITY_RANGE,
                "command_name": TWIST_COMMAND_NAME,
                "command_threshold": 0.1,
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


def _rewards() -> dict[str, RewardTermCfg]:
    """Velocity tracking + two-term minimal gait shaping.

    The categorical converges on a degenerate "tip-toe + forward-lean"
    attractor that satisfies velocity tracking with minimum action
    variance. Two-term minimum surgical shaping addresses the two
    visible symptoms:

      - ``feet_air_time``: rewards step durations in [0.05, 0.5]s.
        Breaks the short-stride tip-toe attractor.
      - ``flat_orientation_l2``: penalises non-upright base orientation.
        Fixes the forward-lean compensation pattern.

    Task weights kept at 2.0 to measure the marginal effect of shaping
    without confounding by a rescale. Other naturalness properties
    (arm swing, smoothness) left to the prior.
    """
    return {
        "track_linear_velocity": RewardTermCfg(
            func=vel_mdp.track_linear_velocity,
            weight=2.0,
            params={"command_name": TWIST_COMMAND_NAME, "std": math.sqrt(0.25)},
        ),
        "track_angular_velocity": RewardTermCfg(
            func=vel_mdp.track_angular_velocity,
            weight=2.0,
            params={"command_name": TWIST_COMMAND_NAME, "std": math.sqrt(0.5)},
        ),
        # Upper-body home-pose regularisation. The categorical policy has no
        # goal for the arms, so PPO recruits them for balance/exploration,
        # producing tremor and erratic upper-body motion on top of an
        # otherwise-decent gait. This anchors shoulders/elbows/wrists to the
        # home pose (the residual baseline, use_default_offset=True) with
        # exp(-mean(err^2/std^2)) -- mjlab's `posture` reward. Weight is 10x
        # below the tracking terms (0.2 vs 2.0) so it silences the arms
        # without competing with velocity tracking. Std values copied from
        # the mjlab g1 velocity task (walking regime, arm joints).
        "upper_body_posture": RewardTermCfg(
            func=vel_mdp.posture,
            weight=0.4,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=(
                        ".*_shoulder_pitch_joint",
                        ".*_shoulder_roll_joint",
                        ".*_shoulder_yaw_joint",
                        ".*_elbow_joint",
                        ".*_wrist_.*",
                    ),
                ),
                "std": {
                    r".*shoulder_pitch.*": 0.15,
                    r".*shoulder_roll.*": 0.15,
                    r".*shoulder_yaw.*": 0.1,
                    r".*elbow.*": 0.15,
                    r".*wrist.*": 0.3,
                },
            },
        ),
        # Penalise robot self-collisions (e.g. an arm intersecting the torso
        # or legs). Returns the count of detected self-contacts; requires the
        # `self_collision` sensor and the FULL_COLLISION preset, both enabled
        # in config/g1/env_cfgs.py. Without those the signal is always zero.
        "self_collisions": RewardTermCfg(
            func=mdp.self_collision_cost,
            weight=-0.5,
            params={"sensor_name": "self_collision"},
        ),
        # Action smoothness penalties to attenuate the standing jitter seen on
        # the real robot. Same terms/weights added to the expert EncDec env:
        # action_rate_l2 penalises the action velocity (||a_t - a_{t-1}||^2),
        # action_acc_l2 the action acceleration / jerk
        # (||a_t - 2 a_{t-1} + a_{t-2}||^2). Both push the categorical toward
        # quieter actions at rest without a standing-specific gate.
        "action_rate_l2": RewardTermCfg(func=mdp.action_rate_l2, weight=-1e-1),
        "action_acc_l2": RewardTermCfg(func=mdp.action_acc_l2, weight=-5e-2),
        # "feet_air_time": RewardTermCfg(
        #     func=vel_mdp.feet_air_time,
        #     weight=1.0,
        #     params={
        #         "sensor_name": "feet_ground_contact",
        #         "threshold_min": 0.05,
        #         "threshold_max": 0.5,
        #         "command_name": TWIST_COMMAND_NAME,
        #         "command_threshold": 0.1,
        #     },
        # ),
        # "flat_orientation_l2": RewardTermCfg(
        #     func=vel_mdp.flat_orientation_l2,
        #     weight=-1.0,
        # ),
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
        "command_vel": CurriculumTermCfg(
            func=vel_mdp.commands_vel,
            params={
                "command_name": TWIST_COMMAND_NAME,
                "velocity_stages": [
                    {
                        "step": 0,
                        "lin_vel_x": (0.0, 0.5),
                        "lin_vel_y": (-0.08, 0.08),
                        "ang_vel_z": (-0.5, 0.5),
                    },
                    {
                        "step": 500 * 24,
                        "lin_vel_x": (-0.2, 0.8),
                        "lin_vel_y": (-0.2, 0.2),
                        "ang_vel_z": (-1.0, 1.0),
                    },
                    {
                        "step": 1000 * 24,
                        "lin_vel_x": (-0.5, 1.5),
                        "lin_vel_y": (-0.4, 0.4),
                        "ang_vel_z": (-1.2, 1.2),
                    },
                    {
                        "step": 2000 * 24,
                        "lin_vel_x": (-0.8, 2.5),
                        "lin_vel_y": (-0.6, 0.6),
                        "ang_vel_z": (-2.0, 2.0),
                    },
                    {
                        "step": 3000 * 24,
                        "lin_vel_x": (-1.0, 3.0),
                        "lin_vel_y": (-1.0, 1.0),
                        "ang_vel_z": (-3.0, 3.0),
                    },
                ],
            },
        ),
    }


def make_locomotion_env_cfg() -> ManagerBasedRlEnvCfg:
    """YAHMP omnidirectional locomotion task template (residual joint actions)."""
    return _make_env_cfg(actions=_actions())


def _make_env_cfg(*, actions: dict[str, ActionTermCfg]) -> ManagerBasedRlEnvCfg:
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
        TWIST_COMMAND_NAME: UniformVelocityCommandCfg(**_velocity_command_kwargs()),
    }

    return ManagerBasedRlEnvCfg(
        scene=SceneCfg(terrain=TerrainEntityCfg(terrain_type="plane"), num_envs=1),
        observations=observations,
        actions=actions,
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
        episode_length_s=10.0,
    )
