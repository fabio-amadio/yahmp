import math
import os
import shutil
from pathlib import Path
from typing import Any

import torch
import wandb
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper

from yahmp.mdp.motion.base import MotionCommand
from yahmp.rl.exporter import attach_onnx_metadata
from yahmp.rl.motion_stats import dump_motion_stats
from yahmp.rl.reward_logging import (
  install_actual_episode_length_reward_logging,
  install_merged_timeout_termination_logging,
  install_yahmp_logger,
)
from yahmp.rl.student_teacher_policy import YahmpStudentTeacherActorModel
from yahmp.utils import get_wandb_checkpoint_path


class YahmpOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    self._configure_model_cfg(env, train_cfg)
    if isinstance(env, RslRlVecEnvWrapper):
      install_actual_episode_length_reward_logging(env)
      install_merged_timeout_termination_logging(env)
    super().__init__(env, train_cfg, log_dir, device)
    install_yahmp_logger(self)
    self.registry_name = registry_name

  def _upload_model_mode(self) -> str:
    mode = str(self.cfg.get("upload_model_mode", "rolling_latest"))
    if mode not in {"all", "rolling_latest"}:
      raise ValueError(
        f"Unsupported upload_model_mode `{mode}`. Expected one of: all, rolling_latest."
      )
    return mode

  def _maybe_upload_checkpoint(self, checkpoint_path: Path) -> None:
    if not self.cfg.get("upload_model", True):
      return

    mode = self._upload_model_mode()
    if mode == "all":
      self.logger.save_model(str(checkpoint_path), self.current_learning_iteration)
      return

    latest_path = checkpoint_path.with_name("model_latest.pt")
    shutil.copy2(checkpoint_path, latest_path)
    self.logger.save_model(str(latest_path), self.current_learning_iteration)

  @staticmethod
  def _infer_motion_dims(
    env: RslRlVecEnvWrapper, representation: str
  ) -> tuple[int, int] | None:
    env_unwrapped = env.unwrapped
    motion_term = env_unwrapped.command_manager.get_term("motion")
    if not isinstance(motion_term, MotionCommand):
      return None
    if not motion_term.has_command_representation(representation):
      return None
    motion_obs = motion_term.get_command_representation(representation)
    motion_obs_dim = int(motion_obs.shape[-1])
    motion_steps = len(motion_term.future_sampling_step_offsets)
    if motion_steps <= 0:
      motion_steps = 1
    return motion_obs_dim, motion_steps

  @staticmethod
  def _obs_group_name(train_cfg: dict, obs_set: str, default: str) -> str:
    obs_groups = train_cfg.get("obs_groups", {})
    group_names = obs_groups.get(obs_set, (default,))
    if isinstance(group_names, str):
      return group_names
    if len(group_names) == 0:
      return default
    return str(group_names[0])

  @classmethod
  def _configure_model_cfg(cls, env, train_cfg: dict) -> None:
    if not isinstance(env, RslRlVecEnvWrapper):
      return

    actor_cfg = train_cfg.get("actor", {})
    critic_cfg = train_cfg.get("critic", {})

    actor_class = str(actor_cfg.get("class_name", ""))
    critic_class = str(critic_cfg.get("class_name", ""))

    motion_dims = cls._infer_motion_dims(env, "default")
    teacher_dims = cls._infer_motion_dims(env, "teacher")

    if actor_class == "yahmp.rl.policy:YahmpActorModel" and motion_dims is not None:
      observation_manager = env.unwrapped.observation_manager
      obs_dims = observation_manager.group_obs_dim
      actor_group = cls._obs_group_name(train_cfg, "actor", "actor")
      if "policy_current" in obs_dims and "policy_history" in obs_dims:
        current_dim = int(obs_dims["policy_current"][0])
        history_dim = int(obs_dims["policy_history"][0])
        command_dim = motion_dims[0] // max(motion_dims[1], 1)
      else:
        actor_terms = observation_manager.active_terms[actor_group]
        actor_term_dims = observation_manager._group_obs_term_dim[actor_group]
        flat_term_dims = {
          name: int(math.prod(dims))
          for name, dims in zip(actor_terms, actor_term_dims, strict=False)
        }
        command_dim = flat_term_dims["command"]
        history_dim = flat_term_dims["history"]
        current_dim = int(obs_dims[actor_group][0]) - command_dim - history_dim
      history_step_dim = command_dim + current_dim
      if history_dim % history_step_dim != 0:
        raise ValueError(
          "YahmpActorModel history dimension mismatch: "
          f"history_dim={history_dim}, history_step_dim={history_step_dim}."
        )
      actor_cfg.setdefault("current_motion_obs_dim", command_dim)
      actor_cfg.setdefault("proprio_obs_dim", current_dim)
      actor_cfg.setdefault("history_steps", max(history_dim // history_step_dim, 1))
      actor_cfg.setdefault("history_latent_dim", 128)
      actor_cfg.setdefault("history_conv_channels", (64, 32))
      actor_cfg.setdefault("history_conv_kernel_sizes", (4, 2))
      actor_cfg.setdefault("history_conv_strides", (2, 1))
      actor_cfg.setdefault("layer_norm", True)

    if (
      actor_class == "yahmp.rl.policy:YahmpFutureActorModel" and motion_dims is not None
    ):
      observation_manager = env.unwrapped.observation_manager
      obs_dims = observation_manager.group_obs_dim
      actor_group = cls._obs_group_name(train_cfg, "actor", "actor")
      actor_terms = observation_manager.active_terms[actor_group]
      actor_term_dims = observation_manager._group_obs_term_dim[actor_group]
      flat_term_dims = {
        name: int(math.prod(dims))
        for name, dims in zip(actor_terms, actor_term_dims, strict=False)
      }
      history_term_cfg = observation_manager._group_obs_term_cfgs[actor_group][
        actor_terms.index("history")
      ]
      history_length = int(
        getattr(history_term_cfg, "params", {}).get("history_length", 1)
      )
      history_dim = flat_term_dims["history"]
      current_dim = int(obs_dims[actor_group][0]) - int(motion_dims[0]) - history_dim
      history_step_dim = flat_term_dims["history"] // max(history_length, 1)
      if history_dim % history_step_dim != 0:
        raise ValueError(
          "YahmpFutureActorModel history dimension mismatch: "
          f"history_dim={history_dim}, history_step_dim={history_step_dim}."
        )
      actor_cfg.setdefault("motion_obs_dim", motion_dims[0])
      actor_cfg.setdefault("motion_steps", motion_dims[1])
      actor_cfg.setdefault("proprio_obs_dim", current_dim)
      actor_cfg.setdefault("history_input_dim", history_step_dim)
      actor_cfg.setdefault("history_steps", max(history_dim // history_step_dim, 1))
      actor_cfg.setdefault("motion_latent_dim", 64)
      actor_cfg.setdefault("history_latent_dim", 128)
      actor_cfg.setdefault("motion_conv_channels", (48, 24))
      actor_cfg.setdefault("motion_conv_kernel_sizes", (6, 4))
      actor_cfg.setdefault("motion_conv_strides", (2, 2))
      actor_cfg.setdefault("history_conv_channels", (64, 32))
      actor_cfg.setdefault("history_conv_kernel_sizes", (4, 2))
      actor_cfg.setdefault("history_conv_strides", (2, 1))
      actor_cfg.setdefault("layer_norm", True)

    if (
      critic_class == "yahmp.rl.policy:YahmpFutureCriticModel"
      and motion_dims is not None
    ):
      critic_cfg.setdefault("motion_obs_dim", motion_dims[0])
      critic_cfg.setdefault("motion_steps", motion_dims[1])
      critic_cfg.setdefault("motion_latent_dim", 64)
      critic_cfg.setdefault("motion_conv_channels", (48, 24))
      critic_cfg.setdefault("motion_conv_kernel_sizes", (6, 4))
      critic_cfg.setdefault("motion_conv_strides", (2, 2))
      critic_cfg.setdefault("layer_norm", True)

    if (
      actor_class == "yahmp.rl.student_teacher_policy:YahmpStudentTeacherActorModel"
      and teacher_dims is not None
    ):
      observation_manager = env.unwrapped.observation_manager
      obs_dims = observation_manager.group_obs_dim
      actor_group = cls._obs_group_name(train_cfg, "actor", "actor")
      teacher_group = cls._obs_group_name(train_cfg, "teacher", "teacher_actor")
      actor_terms = observation_manager.active_terms[actor_group]
      actor_term_dims = observation_manager._group_obs_term_dim[actor_group]
      flat_term_dims = {
        name: int(math.prod(dims))
        for name, dims in zip(actor_terms, actor_term_dims, strict=False)
      }
      command_dim = flat_term_dims["command"]
      history_dim = flat_term_dims["history"]
      current_dim = int(obs_dims[actor_group][0]) - command_dim - history_dim
      history_step_dim = command_dim + current_dim
      if history_dim % history_step_dim != 0:
        raise ValueError(
          "YahmpStudentTeacherActorModel history dimension mismatch: "
          f"history_dim={history_dim}, history_step_dim={history_step_dim}."
        )
      actor_cfg.setdefault("current_motion_obs_dim", command_dim)
      actor_cfg.setdefault("proprio_obs_dim", current_dim)
      actor_cfg.setdefault("history_steps", max(history_dim // history_step_dim, 1))
      actor_cfg.setdefault("history_latent_dim", 128)
      actor_cfg.setdefault("history_conv_channels", (64, 32))
      actor_cfg.setdefault("history_conv_kernel_sizes", (4, 2))
      actor_cfg.setdefault("history_conv_strides", (2, 1))
      actor_cfg.setdefault("layer_norm", True)
      actor_cfg.setdefault(
        "teacher_distribution_cfg",
        {
          "class_name": "GaussianDistribution",
          "init_std": 1.0,
          "std_type": "log",
        },
      )
      teacher_terms = observation_manager.active_terms[teacher_group]
      teacher_term_dims = observation_manager._group_obs_term_dim[teacher_group]
      teacher_flat_term_dims = {
        name: int(math.prod(dims))
        for name, dims in zip(teacher_terms, teacher_term_dims, strict=False)
      }
      teacher_command_dim = teacher_flat_term_dims["command"]
      teacher_history_dim = teacher_flat_term_dims["history"]
      teacher_current_dim = (
        int(obs_dims[teacher_group][0]) - teacher_command_dim - teacher_history_dim
      )
      teacher_motion_steps = max(teacher_command_dim // max(command_dim, 1), 1)
      actor_cfg.setdefault("teacher_motion_obs_dim", teacher_command_dim)
      actor_cfg.setdefault("teacher_motion_steps", teacher_motion_steps)
      actor_cfg.setdefault("teacher_proprio_obs_dim", teacher_current_dim)
      actor_cfg.setdefault("teacher_motion_latent_dim", 64)
      actor_cfg.setdefault("teacher_motion_conv_channels", (48, 24))
      actor_cfg.setdefault("teacher_motion_conv_kernel_sizes", (6, 4))
      actor_cfg.setdefault("teacher_motion_conv_strides", (2, 2))
      actor_cfg.setdefault("teacher_history_latent_dim", 128)
      actor_cfg.setdefault("teacher_history_conv_channels", (64, 32))
      actor_cfg.setdefault("teacher_history_conv_kernel_sizes", (4, 2))
      actor_cfg.setdefault("teacher_history_conv_strides", (2, 1))
      actor_cfg.setdefault("teacher_layer_norm", True)

    train_cfg["actor"] = actor_cfg
    train_cfg["critic"] = critic_cfg

  def save(self, path: str, infos=None) -> None:
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    infos = {**(infos or {}), "env_state": env_state}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = infos
    torch.save(saved_dict, path)
    self._maybe_upload_checkpoint(Path(path))
    dump_motion_stats(
      self.env,
      self.cfg,
      Path(path),
      self.current_learning_iteration,
      logger_type=getattr(self.logger, "logger_type", None),
    )

    policy_path = Path(path).parent
    filename = f"{policy_path.parent.name}.onnx"
    try:
      self.export_policy_to_onnx(str(policy_path), filename)
      run_name = (
        wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
      )
      attach_onnx_metadata(self.env.unwrapped, run_name, str(policy_path / filename))
      if self.logger.logger_type in ["wandb"] and self.cfg["upload_model"]:
        wandb.save(
          str(policy_path / filename), base_path=os.path.dirname(str(policy_path))
        )
        if self.registry_name is not None:
          wandb.run.use_artifact(self.registry_name)  # type: ignore[union-attr]
          self.registry_name = None
    except Exception as exc:
      print(f"[WARN] ONNX export failed (training continues): {exc}")

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)

    actor_sd = loaded_dict.get("actor_state_dict", {})
    _normalize_gaussian_distribution_state_dict(actor_sd)

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if load_iteration and infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos


def _normalize_gaussian_distribution_state_dict(state_dict: dict[str, Any]) -> None:
  if "std" in state_dict:
    state_dict["distribution.std_param"] = state_dict.pop("std")
  if "log_std" in state_dict:
    state_dict["distribution.log_std_param"] = state_dict.pop("log_std")


class YahmpStudentOnPolicyRunner(YahmpOnPolicyRunner):
  """On-policy runner with explicit teacher checkpoint loading for student tasks."""

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ) -> None:
    self._yahmp_log_dir = Path(log_dir).resolve() if log_dir is not None else None
    super().__init__(env, train_cfg, log_dir, device, registry_name=registry_name)
    self._maybe_load_teacher_checkpoint()

  def _teacher_checkpoint_path(self) -> Path | None:
    checkpoint_file = self.cfg.get("teacher_checkpoint_file")
    if checkpoint_file:
      checkpoint_path = Path(str(checkpoint_file)).expanduser().resolve()
      if not checkpoint_path.exists():
        raise FileNotFoundError(f"Teacher checkpoint file not found: {checkpoint_path}")
      print(f"[INFO]: Using local teacher checkpoint: {checkpoint_path}")
      return checkpoint_path

    wandb_run_path = self.cfg.get("teacher_wandb_run_path")
    if not wandb_run_path:
      return None

    if self._yahmp_log_dir is None:
      raise ValueError("Cannot resolve teacher W&B checkpoint without a log directory.")
    log_root_path = self._yahmp_log_dir.parent
    checkpoint_path, was_cached = get_wandb_checkpoint_path(
      log_root_path,
      Path(str(wandb_run_path)),
      self.cfg.get("teacher_wandb_checkpoint_name"),
    )
    run_id = checkpoint_path.parent.name
    checkpoint_name = checkpoint_path.name
    cached_str = "cached" if was_cached else "downloaded"
    print(
      "[INFO]: Resolved teacher checkpoint: "
      f"{checkpoint_name} (run: {run_id}, {cached_str})"
    )
    return checkpoint_path

  def _maybe_load_teacher_checkpoint(self) -> None:
    actor = getattr(self.alg, "actor", None)
    if not isinstance(actor, YahmpStudentTeacherActorModel):
      if any(
        self.cfg.get(key)
        for key in (
          "teacher_wandb_run_path",
          "teacher_wandb_checkpoint_name",
          "teacher_checkpoint_file",
        )
      ):
        raise ValueError(
          "Teacher checkpoint options are only supported for "
          "YahmpStudentTeacherActorModel student runs."
        )
      return

    checkpoint_path = self._teacher_checkpoint_path()
    if checkpoint_path is None:
      return

    loaded_dict = torch.load(
      checkpoint_path, map_location=self.device, weights_only=False
    )
    teacher_state_dict = loaded_dict.get("teacher_state_dict")
    if teacher_state_dict is None:
      teacher_state_dict = loaded_dict.get("actor_state_dict")
    if teacher_state_dict is None:
      raise ValueError(
        "Teacher checkpoint does not contain `teacher_state_dict` or "
        "`actor_state_dict`."
      )
    _normalize_gaussian_distribution_state_dict(teacher_state_dict)
    actor.teacher.load_state_dict(
      teacher_state_dict,
      strict=bool(self.cfg.get("teacher_strict_load", True)),
    )
    actor.loaded_teacher = True
    actor.teacher.eval()

  def load(
    self,
    path: str,
    load_cfg: dict | None = None,
    strict: bool = True,
    map_location: str | None = None,
  ) -> dict:
    loaded_dict = torch.load(path, map_location=map_location, weights_only=False)
    actor_sd = loaded_dict.get("actor_state_dict", {})
    _normalize_gaussian_distribution_state_dict(actor_sd)

    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if load_iteration and infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]

    actor = getattr(self.alg, "actor", None)
    if isinstance(actor, YahmpStudentTeacherActorModel):
      if any(key.startswith("teacher.") for key in actor_sd):
        actor.loaded_teacher = True
        actor.teacher.eval()
    # Allow an explicit teacher checkpoint to override embedded teacher weights
    # from a resumed student checkpoint.
    self._maybe_load_teacher_checkpoint()
    return infos
