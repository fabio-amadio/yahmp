"""YAHMP student task configuration."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from yahmp import mdp
from yahmp.mdp import TeacherStudentJointRefAnchorRpMotionCommandCfg
from yahmp.yahmp_env_cfg import make_yahmp_env_cfg
from yahmp.yahmp_teacher_env_cfg import (
  DEFAULT_TEACHER_FUTURE_STEPS,
  PUSH_VELOCITY_RANGE,
  make_yahmp_teacher_env_cfg,
)


def student_motion_command_kwargs() -> dict[str, object]:
  """Return shared kwargs for the student motion command."""
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
    "motion_file": "",
    "anchor_body_name": "",
    "body_names": (),
    "root_body_name": "",
    "sampling_mode": "adaptive",
  }


def make_teacher_policy_observation_group() -> ObservationGroupCfg:
  """Build the frozen teacher observation group."""
  teacher_cfg = make_yahmp_teacher_env_cfg()
  teacher_policy_group = teacher_cfg.observations["policy"]
  teacher_policy_terms = deepcopy(teacher_policy_group.terms)
  teacher_policy_terms["command"] = ObservationTermCfg(
    func=mdp.motion_teacher_policy_command,
    params={"command_name": "motion"},
  )
  return ObservationGroupCfg(
    terms=teacher_policy_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )


def make_yahmp_student_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the YAHMP student task configuration."""
  cfg = make_yahmp_env_cfg()
  cfg.commands["motion"] = TeacherStudentJointRefAnchorRpMotionCommandCfg(
    **student_motion_command_kwargs(),
    future_sampling_step_offsets=DEFAULT_TEACHER_FUTURE_STEPS,
  )
  cfg.observations["teacher_policy"] = make_teacher_policy_observation_group()

  return cfg
