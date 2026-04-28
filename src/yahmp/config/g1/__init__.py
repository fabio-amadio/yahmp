from mjlab.tasks.registry import register_mjlab_task

from yahmp.rl import YahmpOnPolicyRunner

from .env_cfgs import (
  unitree_g1_yahmp_env_cfg,
  unitree_g1_yahmp_future_env_cfg,
  unitree_g1_yahmp_student_env_cfg,
  unitree_g1_yahmp_teacher_env_cfg,
)
from .rl_cfg import (
  unitree_g1_yahmp_ppo_runner_cfg,
  unitree_g1_yahmp_future_ppo_runner_cfg,
  unitree_g1_yahmp_student_action_matching_rl_runner_cfg,
  unitree_g1_yahmp_student_kl_matching_rl_runner_cfg,
  unitree_g1_yahmp_teacher_ppo_runner_cfg,
)

register_mjlab_task(
  task_id="Mjlab-YAHMP-Unitree-G1",
  env_cfg=unitree_g1_yahmp_env_cfg(),
  play_env_cfg=unitree_g1_yahmp_env_cfg(play=True),
  rl_cfg=unitree_g1_yahmp_ppo_runner_cfg(),
  runner_cls=YahmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-YAHMP-Teacher-Unitree-G1",
  env_cfg=unitree_g1_yahmp_teacher_env_cfg(),
  play_env_cfg=unitree_g1_yahmp_teacher_env_cfg(play=True),
  rl_cfg=unitree_g1_yahmp_teacher_ppo_runner_cfg(),
  runner_cls=YahmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-YAHMP-Future-Unitree-G1",
  env_cfg=unitree_g1_yahmp_future_env_cfg(),
  play_env_cfg=unitree_g1_yahmp_future_env_cfg(play=True),
  rl_cfg=unitree_g1_yahmp_future_ppo_runner_cfg(),
  runner_cls=YahmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-YAHMP-Student-RL+Action-Matching-Unitree-G1",
  env_cfg=unitree_g1_yahmp_student_env_cfg(),
  play_env_cfg=unitree_g1_yahmp_student_env_cfg(play=True),
  rl_cfg=unitree_g1_yahmp_student_action_matching_rl_runner_cfg(),
  runner_cls=YahmpOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-YAHMP-Student-RL+KL-Matching-Unitree-G1",
  env_cfg=unitree_g1_yahmp_student_env_cfg(),
  play_env_cfg=unitree_g1_yahmp_student_env_cfg(play=True),
  rl_cfg=unitree_g1_yahmp_student_kl_matching_rl_runner_cfg(),
  runner_cls=YahmpOnPolicyRunner,
)
