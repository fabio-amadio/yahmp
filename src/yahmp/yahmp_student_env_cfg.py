"""YAHMP student task configuration."""

from copy import deepcopy

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationGroupCfg, ObservationTermCfg

from yahmp import mdp
from yahmp.mdp import TeacherStudentJointRefAnchorRpMotionCommandCfg
from yahmp.yahmp_env_cfg import (
  _yahmp_motion_command_kwargs,
  make_yahmp_env_cfg,
)
from yahmp.yahmp_future_env_cfg import FUTURE_STEPS
from yahmp.yahmp_teacher_env_cfg import make_yahmp_teacher_env_cfg


def student_motion_command_kwargs() -> dict[str, object]:
  """Return shared kwargs for the student motion command."""
  return dict(_yahmp_motion_command_kwargs())


def make_teacher_actor_observation_group() -> ObservationGroupCfg:
  """Build the frozen teacher-actor observation group."""
  teacher_cfg = make_yahmp_teacher_env_cfg()
  teacher_actor_group = teacher_cfg.observations["actor"]
  teacher_actor_terms = deepcopy(teacher_actor_group.terms)
  teacher_actor_terms["command"] = ObservationTermCfg(
    func=mdp.motion_teacher_actor_command,
    params={"command_name": "motion"},
  )
  return ObservationGroupCfg(
    terms=teacher_actor_terms,
    concatenate_terms=True,
    enable_corruption=False,
  )


def make_yahmp_student_env_cfg() -> ManagerBasedRlEnvCfg:
  """Create the YAHMP student task configuration."""
  cfg = make_yahmp_env_cfg()
  cfg.commands["motion"] = TeacherStudentJointRefAnchorRpMotionCommandCfg(
    **student_motion_command_kwargs(),
    future_sampling_step_offsets=FUTURE_STEPS,
  )
  cfg.observations["teacher_actor"] = make_teacher_actor_observation_group()

  return cfg
