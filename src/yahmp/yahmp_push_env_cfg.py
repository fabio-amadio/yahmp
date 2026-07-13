import math

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers import TerminationTermCfg
from mjlab.managers.command_manager import CommandTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp as vel_mdp

from yahmp import mdp
from yahmp.yahmp_locomotion_env_cfg import (
    HISTORY_LENGTH,
    PUSH_VELOCITY_RANGE,
    _privileged_terms,
    _proprio_actor_terms,
    _proprio_critic_terms,
    make_locomotion_env_cfg,
)

PUSH_COMMAND_NAME = "push"


def _push_command_cfg() -> mdp.PushGoalCommandCfg:
    return mdp.PushGoalCommandCfg(
        entity_name="robot",
        resampling_time_range=(20, 30),
        distance_range=(1.0, 2.0),
        angle_range=(-0.8, 0.8),
        reach_tol=0.2,
    )


def _command_term() -> ObservationTermCfg:
    return ObservationTermCfg(
        func=mdp.push_command,
        params={"command_name": PUSH_COMMAND_NAME},
    )


def _history_term(*, include_privileged: bool = False) -> ObservationTermCfg:
    return ObservationTermCfg(
        func=mdp.YahmpLocomotionObservationHistory,
        params={
            "command_name": PUSH_COMMAND_NAME,
            "history_length": HISTORY_LENGTH,
            "include_privileged": include_privileged,
        },
    )


def _rewards() -> dict[str, RewardTermCfg]:
    return {
        "push_velocity_tracking": RewardTermCfg(
            func=mdp.push_velocity_tracking_exp,
            weight=0.5,
            params={"command_name": PUSH_COMMAND_NAME, "std": 0.5, "cruise_speed": 1.1},
        ),
        "push_progress": RewardTermCfg(
            func=mdp.push_progress,
            weight=1.0,
            params={
                "command_name": PUSH_COMMAND_NAME,
                "cap": 2.0,
            },
        ),
        "push_position_tracking": RewardTermCfg(
            func=mdp.push_position_tracking_exp,
            weight=0.5,
            params={"command_name": PUSH_COMMAND_NAME, "std": 0.5},
        ),
        "push_success": RewardTermCfg(
            func=mdp.push_success_bonus,
            weight=50.0,
            params={
                "command_name": PUSH_COMMAND_NAME,
            },
        ),
        "push_hands_on_crate": RewardTermCfg(
            func=mdp.push_hands_on_crate,
            weight=0.4,
            params={
                "command_name": PUSH_COMMAND_NAME,
                "std": 0.4,
                "asset_cfg": SceneEntityCfg("robot", body_names=(".*_wrist_yaw_link",)),
            },
        ),
    }


def _curriculum() -> dict[str, CurriculumTermCfg]:
    return {
        "push_levels": CurriculumTermCfg(
            func=mdp.push_command_levels,
            params={
                "command_name": PUSH_COMMAND_NAME,
                "stages": [
                    {
                        "step": 0,
                        "distance_range": (0.5, 1.0),
                        "angle_range": (-0.4, 0.4),
                    },
                    {
                        "step": 1000 * 24,
                        "distance_range": (0.5, 1.5),
                        "angle_range": (-0.8, 0.8),
                    },
                    {
                        "step": 2500 * 24,
                        "distance_range": (1.0, 2.5),
                        "angle_range": (-1.5, 1.5),
                    },
                    {
                        "step": 5000 * 24,
                        "distance_range": (1.5, 3.0),
                        "angle_range": (-2.0, 2.0),
                    },
                ],
            },
        ),
    }


def make_push_env_cfg() -> ManagerBasedRlEnvCfg:
    """YAHMP continuous point-to-point navigation task template."""
    cfg = make_locomotion_env_cfg()

    cfg.commands = {PUSH_COMMAND_NAME: _push_command_cfg()}

    # Rebuild the obs groups: same proprio/history/privileged, new command term.
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

    cfg.events["push_robot"].func = mdp.push_moving_envs_by_robot_speed
    cfg.events["push_robot"].params = {
        "velocity_range": PUSH_VELOCITY_RANGE,
        "speed_threshold": 0.3,
    }

    cfg.rewards = _rewards()
    cfg.terminations["reached_goal"] = TerminationTermCfg(
        func=mdp.reached_termination,
        params={"command_name": PUSH_COMMAND_NAME},
    )

    # Keep the (fixed, in-front) crate actually in front of the robot: constrain
    # the base reset to a narrow forward cone / small offset instead of the full
    # 360deg yaw randomisation inherited from locomotion.
    cfg.events["reset_base"].params["pose_range"] = {
        "x": (-0.2, 0.2),
        "y": (-0.2, 0.2),
        "z": (0.0, 0.0),
        "yaw": (-0.3, 0.3),
    }
    # Re-place the crate at (init_state + env_origin) on every reset (also
    # applies the per-env grid offset so crates don't stack at the origin).
    cfg.events["reset_crate"] = EventTermCfg(
        func=vel_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("crate"),
        },
    )

    cfg.curriculum = _curriculum()
    cfg.episode_length_s = 20.0

    commands: dict[str, CommandTermCfg] = cfg.commands  # type: ignore[assignment]
    assert PUSH_COMMAND_NAME in commands
    return cfg
