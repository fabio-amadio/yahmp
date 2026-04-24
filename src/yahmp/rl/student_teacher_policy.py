from __future__ import annotations

from collections.abc import Mapping

import torch
from rsl_rl.utils import resolve_callable

from yahmp.rl.policy import YahmpActorModel


class YahmpStudentTeacherActorModel(YahmpActorModel):
  """Student actor with a frozen teacher actor for teacher-regularized PPO."""

  def __init__(
    self,
    obs,
    obs_groups,
    obs_set,
    output_dim,
    hidden_dims=(256, 256, 256),
    activation: str = "elu",
    obs_normalization: bool = False,
    distribution_cfg: dict | None = None,
    current_motion_obs_dim: int = 0,
    proprio_obs_dim: int = 0,
    history_steps: int = 10,
    history_latent_dim: int = 64,
    history_conv_channels: tuple[int, ...] | list[int] = (48, 24),
    history_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    history_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    layer_norm: bool = True,
    teacher_class_name: str = "yahmp.rl.policy:YahmpFutureActorModel",
    teacher_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    teacher_activation: str = "elu",
    teacher_obs_normalization: bool = True,
    teacher_distribution_cfg: dict | None = None,
    teacher_motion_obs_dim: int = 1240,
    teacher_motion_steps: int = 20,
    teacher_proprio_obs_dim: int = 0,
    teacher_motion_latent_dim: int = 64,
    teacher_history_latent_dim: int = 128,
    teacher_motion_conv_channels: tuple[int, ...] | list[int] = (48, 24),
    teacher_motion_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    teacher_motion_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    teacher_history_conv_channels: tuple[int, ...] | list[int] = (64, 32),
    teacher_history_conv_kernel_sizes: tuple[int, ...] | list[int] = (4, 2),
    teacher_history_conv_strides: tuple[int, ...] | list[int] = (2, 1),
    teacher_layer_norm: bool = True,
  ) -> None:
    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
      current_motion_obs_dim=current_motion_obs_dim,
      proprio_obs_dim=proprio_obs_dim,
      history_steps=history_steps,
      history_latent_dim=history_latent_dim,
      history_conv_channels=history_conv_channels,
      history_conv_kernel_sizes=history_conv_kernel_sizes,
      history_conv_strides=history_conv_strides,
      layer_norm=layer_norm,
    )

    teacher_class = resolve_callable(teacher_class_name)
    self.teacher = teacher_class(
      obs=obs,
      obs_groups=obs_groups,
      obs_set="teacher",
      output_dim=output_dim,
      hidden_dims=teacher_hidden_dims,
      activation=teacher_activation,
      obs_normalization=teacher_obs_normalization,
      distribution_cfg=teacher_distribution_cfg,
      motion_obs_dim=teacher_motion_obs_dim,
      motion_steps=teacher_motion_steps,
      proprio_obs_dim=teacher_proprio_obs_dim,
      motion_latent_dim=teacher_motion_latent_dim,
      history_latent_dim=teacher_history_latent_dim,
      motion_conv_channels=teacher_motion_conv_channels,
      motion_conv_kernel_sizes=teacher_motion_conv_kernel_sizes,
      motion_conv_strides=teacher_motion_conv_strides,
      history_conv_channels=teacher_history_conv_channels,
      history_conv_kernel_sizes=teacher_history_conv_kernel_sizes,
      history_conv_strides=teacher_history_conv_strides,
      layer_norm=teacher_layer_norm,
    )
    for param in self.teacher.parameters():
      param.requires_grad = False
    self.teacher.eval()
    self.loaded_teacher = False

  def teacher_forward(self, obs) -> torch.Tensor:
    return self.teacher(obs)

  def teacher_distribution_params(self, obs) -> tuple[torch.Tensor, ...]:
    if self.teacher.distribution is None:
      raise ValueError("Teacher policy does not define an output distribution.")
    latent = self.teacher.get_latent(obs)
    mlp_output = self.teacher.mlp(latent)
    self.teacher.distribution.update(mlp_output)
    return self.teacher.output_distribution_params

  def load_distillation_state(
    self,
    student_state_dict: Mapping[str, torch.Tensor],
    teacher_state_dict: Mapping[str, torch.Tensor],
    strict: bool = True,
  ) -> None:
    super().load_state_dict(student_state_dict, strict=strict)
    self.teacher.load_state_dict(teacher_state_dict, strict=strict)
    self.loaded_teacher = True
    self.teacher.eval()

  def train(self, mode: bool = True):
    super().train(mode)
    self.teacher.eval()
    return self
