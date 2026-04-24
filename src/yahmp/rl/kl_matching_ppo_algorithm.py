from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.algorithms import PPO

from yahmp.rl.student_teacher_policy import YahmpStudentTeacherActorModel


class YahmpKlMatchingPPO(PPO):
  """PPO with an additional KL-matching loss on the student actor."""

  actor: YahmpStudentTeacherActorModel

  def __init__(
    self,
    actor,
    critic,
    storage,
    *args,
    kl_coef: float = 0.1,
    kl_coef_min: float = 0.0,
    kl_coef_anneal_iters: int = 10_000,
    **kwargs,
  ) -> None:
    super().__init__(actor, critic, storage, *args, **kwargs)
    self.kl_coef_start = float(kl_coef)
    self.kl_coef_min = float(kl_coef_min)
    self.kl_coef_anneal_iters = int(kl_coef_anneal_iters)
    self.num_kl_updates = 0

  def _current_kl_coef(self) -> float:
    if self.kl_coef_anneal_iters <= 0:
      return self.kl_coef_min
    progress = min(
      max(self.num_kl_updates / float(self.kl_coef_anneal_iters), 0.0), 1.0
    )
    return self.kl_coef_start + (self.kl_coef_min - self.kl_coef_start) * progress

  def update(self) -> dict[str, float]:  # noqa: C901
    if self.rnd is not None or self.symmetry is not None:
      raise NotImplementedError(
        "YahmpKlMatchingPPO currently supports YAHMP PPO setup without RND/symmetry."
      )

    kl_coef = self._current_kl_coef()
    if kl_coef > 0.0 and not self.actor.loaded_teacher:
      raise ValueError(
        "Teacher weights are not loaded. Resume from a YAHMP distillation "
        "checkpoint (or a KL-matching RL checkpoint) before running KlMatchingPPO."
      )

    mean_value_loss = 0.0
    mean_surrogate_loss = 0.0
    mean_entropy = 0.0
    mean_kl_loss = 0.0

    if self.actor.is_recurrent or self.critic.is_recurrent:
      generator = self.storage.recurrent_mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )
    else:
      generator = self.storage.mini_batch_generator(
        self.num_mini_batches, self.num_learning_epochs
      )

    for batch in generator:
      if self.normalize_advantage_per_mini_batch:
        with torch.no_grad():
          batch.advantages = (batch.advantages - batch.advantages.mean()) / (
            batch.advantages.std() + 1e-8
          )

      self.actor(
        batch.observations,
        masks=batch.masks,
        hidden_state=batch.hidden_states[0],
        stochastic_output=True,
      )
      actions_log_prob = self.actor.get_output_log_prob(batch.actions)
      values = self.critic(
        batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1]
      )
      distribution_params = self.actor.output_distribution_params
      entropy = self.actor.output_entropy

      if self.desired_kl is not None and self.schedule == "adaptive":
        with torch.inference_mode():
          kl = self.actor.get_kl_divergence(
            batch.old_distribution_params, distribution_params
          )
          kl_mean = torch.mean(kl)

          if self.is_multi_gpu:
            torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
            kl_mean /= self.gpu_world_size

          if self.gpu_global_rank == 0:
            if kl_mean > self.desired_kl * 2.0:
              self.learning_rate = max(1e-5, self.learning_rate / 1.5)
            elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
              self.learning_rate = min(1e-2, self.learning_rate * 1.5)

          if self.is_multi_gpu:
            lr_tensor = torch.tensor(self.learning_rate, device=self.device)
            torch.distributed.broadcast(lr_tensor, src=0)
            self.learning_rate = lr_tensor.item()

          for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.learning_rate

      ratio = torch.exp(actions_log_prob - torch.squeeze(batch.old_actions_log_prob))
      surrogate = -torch.squeeze(batch.advantages) * ratio
      surrogate_clipped = -torch.squeeze(batch.advantages) * torch.clamp(
        ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
      )
      surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

      if self.use_clipped_value_loss:
        value_clipped = batch.values + (values - batch.values).clamp(
          -self.clip_param, self.clip_param
        )
        value_losses = (values - batch.returns).pow(2)
        value_losses_clipped = (value_clipped - batch.returns).pow(2)
        value_loss = torch.max(value_losses, value_losses_clipped).mean()
      else:
        value_loss = (batch.returns - values).pow(2).mean()

      if kl_coef > 0.0:
        with torch.no_grad():
          teacher_distribution_params = self.actor.teacher_distribution_params(
            batch.observations
          )
        kl_loss = self.actor.get_kl_divergence(
          distribution_params, teacher_distribution_params
        ).mean()
      else:
        kl_loss = torch.zeros((), device=self.device)

      loss = (
        surrogate_loss
        + self.value_loss_coef * value_loss
        - self.entropy_coef * entropy.mean()
        + kl_coef * kl_loss
      )

      self.optimizer.zero_grad()
      loss.backward()

      if self.is_multi_gpu:
        self.reduce_parameters()

      nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
      nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
      self.optimizer.step()

      mean_value_loss += value_loss.item()
      mean_surrogate_loss += surrogate_loss.item()
      mean_entropy += entropy.mean().item()
      mean_kl_loss += kl_loss.item()

    num_updates = self.num_learning_epochs * self.num_mini_batches
    mean_value_loss /= num_updates
    mean_surrogate_loss /= num_updates
    mean_entropy /= num_updates
    mean_kl_loss /= num_updates

    self.storage.clear()
    self.num_kl_updates += 1

    return {
      "value": mean_value_loss,
      "surrogate": mean_surrogate_loss,
      "entropy": mean_entropy,
      "kl": mean_kl_loss,
      "kl_coef": kl_coef,
    }

  def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
    if (
      "student_state_dict" in loaded_dict
      and "teacher_state_dict" in loaded_dict
      and "actor_state_dict" not in loaded_dict
    ):
      if load_cfg is None:
        load_cfg = {
          "actor": True,
          "critic": False,
          "optimizer": False,
          "iteration": False,
          "rnd": False,
        }
      if load_cfg.get("actor"):
        self.actor.load_distillation_state(
          loaded_dict["student_state_dict"],
          loaded_dict["teacher_state_dict"],
          strict=strict,
        )
      return load_cfg.get("iteration", False)

    return super().load(loaded_dict, load_cfg, strict)
