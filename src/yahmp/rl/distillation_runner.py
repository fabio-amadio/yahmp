import os
import shutil
from pathlib import Path

import torch
import wandb
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.runners import DistillationRunner

from yahmp.mdp.motion.base import MotionCommand
from yahmp.rl.exporter import attach_onnx_metadata
from yahmp.rl.motion_stats import dump_motion_stats
from yahmp.rl.reward_logging import (
  install_actual_episode_length_reward_logging,
  install_merged_timeout_termination_logging,
  install_yahmp_logger,
)


class YahmpDistillationRunner(DistillationRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    self._configure_model_cfg(env, train_cfg)
    for key in ("student", "teacher"):
      if key in train_cfg:
        for opt in ("cnn_cfg", "distribution_cfg"):
          if train_cfg[key].get(opt) is None:
            train_cfg[key].pop(opt, None)
    if isinstance(env, RslRlVecEnvWrapper):
      install_actual_episode_length_reward_logging(env)
      install_merged_timeout_termination_logging(env)
    super().__init__(env, train_cfg, log_dir, device)
    install_yahmp_logger(self)

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
  def _configure_model_cfg(env, train_cfg: dict) -> None:
    if not isinstance(env, RslRlVecEnvWrapper):
      return
    env_unwrapped = env.unwrapped
    motion_term = env_unwrapped.command_manager.get_term("motion")
    if not isinstance(motion_term, MotionCommand):
      return
    if not motion_term.has_command_representation("teacher"):
      return

    teacher_cfg = train_cfg.get("teacher", {})
    teacher_class = str(teacher_cfg.get("class_name", ""))
    if teacher_class != "yahmp.rl.policy:YahmpFutureActorModel":
      return

    teacher_motion = motion_term.get_command_representation("teacher")
    teacher_cfg.setdefault("motion_obs_dim", int(teacher_motion.shape[-1]))
    teacher_cfg.setdefault(
      "motion_steps",
      max(len(motion_term.future_sampling_step_offsets), 1),
    )
    teacher_cfg.setdefault("motion_latent_dim", 64)
    teacher_cfg.setdefault("motion_conv_channels", (48, 24))
    teacher_cfg.setdefault("motion_conv_kernel_sizes", (6, 4))
    teacher_cfg.setdefault("motion_conv_strides", (2, 2))
    teacher_cfg.setdefault("layer_norm", True)
    train_cfg["teacher"] = teacher_cfg

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
    load_iteration = self.alg.load(loaded_dict, load_cfg, strict)
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]

    infos = loaded_dict["infos"]
    if load_iteration and infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    return infos
