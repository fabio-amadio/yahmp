from __future__ import annotations

import statistics
from collections import deque
from typing import TYPE_CHECKING

import torch
from mjlab.managers.reward_manager import RewardManager
from mjlab.managers.termination_manager import TerminationManager
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.utils.logger import Logger

if TYPE_CHECKING:
  from rsl_rl.runners import OnPolicyRunner


class YahmpRewardManager(RewardManager):
  """Reward manager that preserves default logs and adds reward-rate views."""

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    if env_ids is None:
      env_ids = slice(None)

    episode_lengths_s = (
      self._env.episode_length_buf[env_ids].to(torch.float32) * self._env.step_dt
    )
    valid_lengths = episode_lengths_s > 0.0
    episode_sums = {
      key: self._episode_sums[key][env_ids].clone() for key in self._episode_sums.keys()
    }

    extras = dict(super().reset(env_ids))
    for key, episodic_sums in episode_sums.items():
      normalized = torch.zeros_like(episodic_sums)
      normalized[valid_lengths] = (
        episodic_sums[valid_lengths] / episode_lengths_s[valid_lengths]
      )
      extras["Reward_per_Sec/" + key] = torch.mean(normalized)
    return extras


class YahmpTerminationManager(TerminationManager):
  """Termination manager that preserves default logs and adds fractions."""

  def reset(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    if env_ids is None:
      env_ids = slice(None)

    term_dones = {
      key: value[env_ids].reshape(-1).clone() for key, value in self._term_dones.items()
    }

    extras = dict(super().reset(env_ids))
    if term_dones:
      ref_mask = next(iter(term_dones.values()))
      timeout_mask = torch.zeros_like(ref_mask, dtype=torch.bool)
      fail_mask = torch.zeros_like(ref_mask, dtype=torch.bool)

      for key, mask in term_dones.items():
        if key in {"time_out", "motion_ref_expired"}:
          timeout_mask |= mask
        else:
          fail_mask |= mask
          extras[f"Termination_Frac/{key}"] = mask.to(torch.float32)

      # Give failure precedence so the timeout/fail fractions stay complementary.
      timeout_mask &= ~fail_mask
      extras["Termination_Frac/time_out"] = timeout_mask.to(torch.float32)
      extras["Termination_Frac/fail"] = fail_mask.to(torch.float32)

    return extras


class YahmpLogger(Logger):
  """Logger that adds reward-rate metrics alongside the default episode stats."""

  def __init__(self, *args, step_dt: float, **kwargs) -> None:
    super().__init__(*args, **kwargs)
    self.step_dt = float(step_dt)
    self.reward_rate_buffer = deque(maxlen=100)

  def process_env_step(
    self,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    extras: dict,
    intrinsic_rewards: torch.Tensor | None = None,
  ) -> None:
    if self.writer is None:
      return

    if "episode" in extras:
      self.ep_extras.append(extras["episode"])
    elif "log" in extras:
      self.ep_extras.append(extras["log"])

    if intrinsic_rewards is not None:
      self.cur_ereward_sum += rewards
      self.cur_ireward_sum += intrinsic_rewards
      self.cur_reward_sum += rewards + intrinsic_rewards
    else:
      self.cur_reward_sum += rewards
    self.cur_episode_length += 1

    new_ids = (dones > 0).nonzero(as_tuple=False)
    if len(new_ids) > 0:
      episode_lengths_s = self.cur_episode_length[new_ids][:, 0] * self.step_dt
      reward_per_second = self.cur_reward_sum[new_ids][:, 0] / torch.clamp(
        episode_lengths_s, min=1.0e-8
      )
      self.reward_rate_buffer.extend(reward_per_second.cpu().numpy().tolist())

    self.rewbuffer.extend(self.cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
    self.lenbuffer.extend(self.cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
    self.cur_reward_sum[new_ids] = 0
    self.cur_episode_length[new_ids] = 0
    if intrinsic_rewards is not None:
      self.erewbuffer.extend(self.cur_ereward_sum[new_ids][:, 0].cpu().numpy().tolist())
      self.irewbuffer.extend(self.cur_ireward_sum[new_ids][:, 0].cpu().numpy().tolist())
      self.cur_ereward_sum[new_ids] = 0
      self.cur_ireward_sum[new_ids] = 0

  def log(self, *args, **kwargs) -> None:
    it = int(kwargs.get("it", args[0] if args else 0))
    super().log(*args, **kwargs)

    if self.writer is None:
      return

    if len(self.reward_rate_buffer) == 0:
      return

    mean_reward_per_second = statistics.mean(self.reward_rate_buffer)
    self.writer.add_scalar("Train/mean_reward_per_second", mean_reward_per_second, it)
    if getattr(self, "logger_type", None) != "wandb":
      self.writer.add_scalar(
        "Train/mean_reward_per_second/time",
        mean_reward_per_second,
        int(self.tot_time),
      )


def install_actual_episode_length_reward_logging(env: RslRlVecEnvWrapper) -> None:
  """Replace the env reward manager with a YAHMP variant adding Reward_per_Sec."""
  env_unwrapped = env.unwrapped
  env_unwrapped.reward_manager = YahmpRewardManager(
    env_unwrapped.cfg.rewards,
    env_unwrapped,
    scale_by_dt=env_unwrapped.cfg.scale_rewards_by_dt,
  )


def install_merged_timeout_termination_logging(env: RslRlVecEnvWrapper) -> None:
  """Replace the env termination manager with a YAHMP variant adding fractions."""
  env_unwrapped = env.unwrapped
  env_unwrapped.termination_manager = YahmpTerminationManager(
    env_unwrapped.cfg.terminations,
    env_unwrapped,
  )


def install_yahmp_logger(runner: OnPolicyRunner) -> None:
  """Replace the runner logger with a YAHMP variant that logs reward rate."""
  old_logger = runner.logger
  new_logger = YahmpLogger(
    log_dir=old_logger.log_dir,
    cfg=old_logger.cfg,
    env_cfg=old_logger.env_cfg,
    num_envs=old_logger.num_envs,
    is_distributed=runner.is_distributed,
    gpu_world_size=runner.gpu_world_size,
    gpu_global_rank=runner.gpu_global_rank,
    device=runner.device,
    step_dt=runner.env.unwrapped.step_dt,
  )
  new_logger.git_status_repos = old_logger.git_status_repos
  runner.logger = new_logger
