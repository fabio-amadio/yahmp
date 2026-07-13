from mjlab.envs import ManagerBasedRlEnvCfg
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
    _privileged_terms,
    _proprio_actor_terms,
    _proprio_critic_terms,
    make_locomotion_env_cfg,
)

PUSH_COMMAND_NAME = "push"


def _push_command_cfg() -> mdp.PushGoalCommandCfg:
    # resampling_time ~inf: goals are resampled ON REACH (multi-goal), never by
    # the timer — a mid-push timer expiry would teleport the goal with no success.
    # distance/angle ranges match curriculum stage 0 (they're overwritten by
    # push_command_levels from the very first reset, `>=` guard).
    return mdp.PushGoalCommandCfg(
        entity_name="robot",
        resampling_time_range=(1e9, 1e9),
        distance_range=(0.5, 1.0),
        angle_range=(-0.4, 0.4),
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
        "push_speed_limit": RewardTermCfg(
            func=mdp.push_speed_limit,
            weight=0.5,
            params={"command_name": PUSH_COMMAND_NAME, "v_max": 1.2},
        ),
        "push_progress": RewardTermCfg(
            func=mdp.push_progress,
            weight=2.0,  # task motor: ~2x distance pushed per episode
            params={
                "command_name": PUSH_COMMAND_NAME,
                "cap": 2.0,
            },
        ),
        "push_position_tracking": RewardTermCfg(
            func=mdp.push_position_tracking_exp,
            weight=0.1,  # last-20cm guide only; higher makes parking profitable
            params={"command_name": PUSH_COMMAND_NAME, "std": 0.5},
        ),
        "push_success": RewardTermCfg(
            func=mdp.push_success_bonus,
            weight=100.0,  # 100*dt=2.0/goal: completing >> parking (x13 margin)
            params={
                "command_name": PUSH_COMMAND_NAME,
            },
        ),
        "push_hands_on_crate": RewardTermCfg(
            func=mdp.push_hands_on_crate,
            weight=0.5,  # bootstrap shaping; at 2.0 hand-gluing dwarfed the task
            params={"command_name": PUSH_COMMAND_NAME, "std": 0.35},
        ),
        "push_hands_contact": RewardTermCfg(
            func=mdp.push_hands_contact,
            weight=0.25,  # nudge + wandb metric; overlaps push_hands_on_crate
            params={"sensor_name": "hands_crate_contact"},
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

    # No external push perturbations on the robot in this task: random shoves
    # would knock the hands off the crate mid-push and fight the hand-gate.
    cfg.events.pop("push_robot", None)

    cfg.rewards = _rewards()
    # MULTI-GOAL: no reached_goal termination — reaching resamples the next goal
    # in-episode (see PushGoalCommand._update_command); episodes end on
    # time_out/fell_over only.

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
    # Longer episodes so each one holds several goal cycles (push, reach,
    # re-position around the crate, push again).
    cfg.episode_length_s = 30.0

    commands: dict[str, CommandTermCfg] = cfg.commands  # type: ignore[assignment]
    assert PUSH_COMMAND_NAME in commands
    return cfg
