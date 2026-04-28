from __future__ import annotations

import copy
from functools import reduce

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict


class MotionEncoder(nn.Module):
  """Temporal encoder for stacked observations."""

  @staticmethod
  def infer_conv_out_dim(
    num_steps: int,
    conv_channels: tuple[int, ...] | list[int],
    conv_kernel_sizes: tuple[int, ...] | list[int],
    conv_strides: tuple[int, ...] | list[int],
  ) -> int:
    conv_channels = tuple(int(v) for v in conv_channels)
    conv_kernel_sizes = tuple(int(v) for v in conv_kernel_sizes)
    conv_strides = tuple(int(v) for v in conv_strides)

    conv_out_length = int(num_steps)
    for kernel, stride in zip(conv_kernel_sizes, conv_strides, strict=True):
      conv_out_length = (conv_out_length - kernel) // stride + 1
      if conv_out_length <= 0:
        raise ValueError(
          "Invalid temporal conv config for given `num_steps`: "
          f"{num_steps}, kernels={conv_kernel_sizes}, strides={conv_strides}."
        )
    return int(conv_channels[-1] * conv_out_length)

  def __init__(
    self,
    input_dim_per_step: int,
    num_steps: int,
    activation: str = "elu",
    conv_channels: tuple[int, ...] | list[int] = (48, 24),
    conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    conv_strides: tuple[int, ...] | list[int] = (2, 2),
    projection_dim: int = 64,
  ) -> None:
    super().__init__()

    self.num_steps = int(num_steps)
    conv_channels = tuple(int(v) for v in conv_channels)
    conv_kernel_sizes = tuple(int(v) for v in conv_kernel_sizes)
    conv_strides = tuple(int(v) for v in conv_strides)
    self.out_dim = int(projection_dim)

    if self.num_steps <= 0:
      raise ValueError(f"`num_steps` must be positive, got {self.num_steps}.")
    if input_dim_per_step <= 0:
      raise ValueError(
        f"`input_dim_per_step` must be positive, got {input_dim_per_step}."
      )
    if self.out_dim <= 0:
      raise ValueError(f"`projection_dim` must be positive, got {self.out_dim}.")
    if len(conv_channels) == 0:
      raise ValueError("`conv_channels` must contain at least one value.")
    if not (len(conv_channels) == len(conv_kernel_sizes) == len(conv_strides)):
      raise ValueError("Conv config lengths must match.")

    conv_layers: list[nn.Module] = []
    in_channels = int(input_dim_per_step)
    for out_channels, kernel, stride in zip(
      conv_channels, conv_kernel_sizes, conv_strides, strict=True
    ):
      conv_layers.append(
        nn.Conv1d(in_channels, out_channels, kernel_size=kernel, stride=stride)
      )
      conv_layers.append(resolve_nn_activation(activation))
      in_channels = out_channels
    self.conv = nn.Sequential(*conv_layers)
    self.flatten = nn.Flatten()

    self.conv_out_dim = self.infer_conv_out_dim(
      num_steps=self.num_steps,
      conv_channels=conv_channels,
      conv_kernel_sizes=conv_kernel_sizes,
      conv_strides=conv_strides,
    )
    self.proj = nn.Sequential(
      nn.Linear(self.conv_out_dim, self.out_dim),
      resolve_nn_activation(activation),
    )

  def forward(self, motion_obs: torch.Tensor) -> torch.Tensor:
    batch_size = motion_obs.shape[0]
    step_obs = motion_obs.reshape(batch_size, self.num_steps, -1).permute(0, 2, 1)
    return self.proj(self.flatten(self.conv(step_obs)))


def _build_mlp(
  input_dim: int,
  output_dim: int | tuple[int, ...] | list[int],
  hidden_dims: tuple[int, ...] | list[int],
  activation: str,
  layer_norm: bool,
) -> nn.Sequential:
  if len(hidden_dims) == 0:
    raise ValueError("`hidden_dims` must contain at least one element.")

  def new_activation() -> nn.Module:
    return resolve_nn_activation(activation)

  layers: list[nn.Module] = []
  in_dim = input_dim
  for idx, hidden_dim in enumerate(hidden_dims):
    layers.append(nn.Linear(in_dim, hidden_dim))
    if layer_norm and len(hidden_dims) > 1 and idx == len(hidden_dims) - 1:
      layers.append(nn.LayerNorm(hidden_dim))
    layers.append(new_activation())
    in_dim = hidden_dim

  if isinstance(output_dim, int):
    layers.append(nn.Linear(in_dim, output_dim))
  else:
    total_out = reduce(lambda x, y: x * y, output_dim)
    layers.append(nn.Linear(in_dim, total_out))
    layers.append(nn.Unflatten(dim=-1, unflattened_size=output_dim))

  return nn.Sequential(*layers)


class YahmpActorModel(MLPModel):
  """Actor with a history encoder."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    current_motion_obs_dim: int = 0,
    proprio_obs_dim: int = 0,
    history_steps: int = 10,
    history_latent_dim: int = 64,
    history_conv_channels: tuple[int, ...] | list[int] = (48, 24),
    history_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    history_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    layer_norm: bool = True,
  ) -> None:
    self.current_motion_obs_dim = int(current_motion_obs_dim)
    self.proprio_obs_dim = int(proprio_obs_dim)
    self.history_steps = int(history_steps)
    self.history_latent_dim = int(history_latent_dim)
    self.layer_norm = bool(layer_norm)

    if self.current_motion_obs_dim <= 0:
      raise ValueError(
        f"`current_motion_obs_dim` must be positive, got {self.current_motion_obs_dim}."
      )
    if self.proprio_obs_dim <= 0:
      raise ValueError(
        f"`proprio_obs_dim` must be positive, got {self.proprio_obs_dim}."
      )
    if self.history_steps <= 0:
      raise ValueError(f"`history_steps` must be positive, got {self.history_steps}.")

    self.current_obs_dim = self.current_motion_obs_dim + self.proprio_obs_dim
    self.history_obs_dim = self.current_obs_dim * self.history_steps

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    expected_obs_dim = self.current_obs_dim + self.history_obs_dim
    if self.obs_dim != expected_obs_dim:
      raise ValueError(
        "YahmpActorModel observation dimension mismatch: "
        f"got {self.obs_dim}, expected {expected_obs_dim} "
        f"({self.current_obs_dim} current + {self.history_obs_dim} history)."
      )

    self.history_encoder = MotionEncoder(
      input_dim_per_step=self.current_obs_dim,
      num_steps=self.history_steps,
      activation=activation,
      conv_channels=history_conv_channels,
      conv_kernel_sizes=history_conv_kernel_sizes,
      conv_strides=history_conv_strides,
      projection_dim=self.history_latent_dim,
    )

    mlp_output_dim = (
      self.distribution.input_dim if self.distribution is not None else output_dim
    )
    self.mlp = _build_mlp(
      input_dim=self._get_latent_dim(),
      output_dim=mlp_output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      layer_norm=self.layer_norm,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def _get_latent_dim(self) -> int:
    return self.current_obs_dim + self.history_latent_dim

  def _split_obs(
    self, obs_flat: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    motion_end = self.current_motion_obs_dim
    current_end = self.current_obs_dim
    history_end = current_end + self.history_obs_dim
    return (
      obs_flat[:, :motion_end],
      obs_flat[:, motion_end:current_end],
      obs_flat[:, current_end:history_end],
    )

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> torch.Tensor:
    obs_flat = super().get_latent(obs, masks, hidden_state)
    motion_obs, proprio_obs, history_obs = self._split_obs(obs_flat)
    history_latent = self.history_encoder(history_obs)
    return torch.cat((motion_obs, proprio_obs, history_latent), dim=-1)

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxYahmpActorModel(self, verbose=verbose)


class YahmpFutureActorModel(MLPModel):
  """Actor with future-motion and history encoders."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    motion_obs_dim: int = 0,
    motion_steps: int = 1,
    proprio_obs_dim: int = 0,
    history_input_dim: int | None = None,
    history_steps: int = 10,
    motion_latent_dim: int = 64,
    history_latent_dim: int = 64,
    motion_conv_channels: tuple[int, ...] | list[int] = (48, 24),
    motion_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    motion_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    history_conv_channels: tuple[int, ...] | list[int] = (48, 24),
    history_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    history_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    layer_norm: bool = True,
  ) -> None:
    self.motion_obs_dim = int(motion_obs_dim)
    self.motion_steps = int(motion_steps)
    self.proprio_obs_dim = int(proprio_obs_dim)
    self.history_input_dim = (
      int(history_input_dim) if history_input_dim is not None else None
    )
    self.history_steps = int(history_steps)
    self.motion_latent_dim = int(motion_latent_dim)
    self.history_latent_dim = int(history_latent_dim)
    self.layer_norm = bool(layer_norm)

    if self.motion_obs_dim <= 0:
      raise ValueError(f"`motion_obs_dim` must be positive, got {self.motion_obs_dim}.")
    if self.motion_steps <= 0:
      raise ValueError(f"`motion_steps` must be positive, got {self.motion_steps}.")
    if self.proprio_obs_dim <= 0:
      raise ValueError(
        f"`proprio_obs_dim` must be positive, got {self.proprio_obs_dim}."
      )
    if self.history_input_dim is not None and self.history_input_dim <= 0:
      raise ValueError(
        "`history_input_dim` must be positive when provided, got "
        f"{self.history_input_dim}."
      )
    if self.history_steps <= 0:
      raise ValueError(f"`history_steps` must be positive, got {self.history_steps}.")
    if self.motion_obs_dim % self.motion_steps != 0:
      raise ValueError(
        "Motion observation dimension must be divisible by motion_steps: "
        f"{self.motion_obs_dim} % {self.motion_steps} != 0."
      )

    self.single_motion_obs_dim = self.motion_obs_dim // self.motion_steps
    self.current_obs_dim = self.single_motion_obs_dim + self.proprio_obs_dim
    self.history_input_dim = (
      self.current_obs_dim if self.history_input_dim is None else self.history_input_dim
    )
    self.history_obs_dim = self.history_input_dim * self.history_steps

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    expected_obs_dim = self.motion_obs_dim + self.proprio_obs_dim + self.history_obs_dim
    if self.obs_dim != expected_obs_dim:
      raise ValueError(
        "YahmpFutureActorModel observation dimension mismatch: "
        f"got {self.obs_dim}, expected {expected_obs_dim} "
        f"({self.motion_obs_dim} motion + {self.proprio_obs_dim} proprio + "
        f"{self.history_obs_dim} history)."
      )

    self.motion_encoder = MotionEncoder(
      input_dim_per_step=self.single_motion_obs_dim,
      num_steps=self.motion_steps,
      activation=activation,
      conv_channels=motion_conv_channels,
      conv_kernel_sizes=motion_conv_kernel_sizes,
      conv_strides=motion_conv_strides,
      projection_dim=self.motion_latent_dim,
    )
    self.history_encoder = MotionEncoder(
      input_dim_per_step=self.history_input_dim,
      num_steps=self.history_steps,
      activation=activation,
      conv_channels=history_conv_channels,
      conv_kernel_sizes=history_conv_kernel_sizes,
      conv_strides=history_conv_strides,
      projection_dim=self.history_latent_dim,
    )

    mlp_output_dim = (
      self.distribution.input_dim if self.distribution is not None else output_dim
    )
    self.mlp = _build_mlp(
      input_dim=self._get_latent_dim(),
      output_dim=mlp_output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      layer_norm=self.layer_norm,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def _get_latent_dim(self) -> int:
    return self.current_obs_dim + self.motion_latent_dim + self.history_latent_dim

  def _split_obs(
    self, obs_flat: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    motion_end = self.motion_obs_dim
    proprio_end = motion_end + self.proprio_obs_dim
    history_end = proprio_end + self.history_obs_dim
    return (
      obs_flat[:, :motion_end],
      obs_flat[:, motion_end:proprio_end],
      obs_flat[:, proprio_end:history_end],
    )

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> torch.Tensor:
    obs_flat = super().get_latent(obs, masks, hidden_state)
    motion_obs, proprio_obs, history_obs = self._split_obs(obs_flat)
    current_motion_obs = motion_obs[:, : self.single_motion_obs_dim]
    motion_latent = self.motion_encoder(motion_obs)
    history_latent = self.history_encoder(history_obs)
    return torch.cat(
      (current_motion_obs, proprio_obs, motion_latent, history_latent), dim=-1
    )

  def as_onnx(self, verbose: bool = False) -> nn.Module:
    return _OnnxYahmpFutureActorModel(self, verbose=verbose)


class YahmpFutureCriticModel(MLPModel):
  """Critic with a future-motion encoder."""

  def __init__(
    self,
    obs: TensorDict,
    obs_groups: dict[str, list[str]],
    obs_set: str,
    output_dim: int,
    hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
    activation: str = "elu",
    obs_normalization: bool = True,
    distribution_cfg: dict | None = None,
    motion_obs_dim: int = 0,
    motion_steps: int = 1,
    motion_latent_dim: int = 64,
    motion_conv_channels: tuple[int, ...] | list[int] = (48, 24),
    motion_conv_kernel_sizes: tuple[int, ...] | list[int] = (6, 4),
    motion_conv_strides: tuple[int, ...] | list[int] = (2, 2),
    layer_norm: bool = True,
  ) -> None:
    self.motion_obs_dim = int(motion_obs_dim)
    self.motion_steps = int(motion_steps)
    self.motion_latent_dim = int(motion_latent_dim)
    self.layer_norm = bool(layer_norm)

    if self.motion_obs_dim <= 0:
      raise ValueError(f"`motion_obs_dim` must be positive, got {self.motion_obs_dim}.")
    if self.motion_steps <= 0:
      raise ValueError(f"`motion_steps` must be positive, got {self.motion_steps}.")
    if self.motion_obs_dim % self.motion_steps != 0:
      raise ValueError(
        "Motion observation dimension must be divisible by motion_steps: "
        f"{self.motion_obs_dim} % {self.motion_steps} != 0."
      )

    self.single_motion_obs_dim = self.motion_obs_dim // self.motion_steps

    super().__init__(
      obs=obs,
      obs_groups=obs_groups,
      obs_set=obs_set,
      output_dim=output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      obs_normalization=obs_normalization,
      distribution_cfg=distribution_cfg,
    )

    if self.obs_dim <= self.motion_obs_dim:
      raise ValueError(
        "YahmpFutureCriticModel expects privileged critic observations after "
        f"the future motion block, got obs_dim={self.obs_dim}, "
        f"motion_obs_dim={self.motion_obs_dim}."
      )

    self.motion_encoder = MotionEncoder(
      input_dim_per_step=self.single_motion_obs_dim,
      num_steps=self.motion_steps,
      activation=activation,
      conv_channels=motion_conv_channels,
      conv_kernel_sizes=motion_conv_kernel_sizes,
      conv_strides=motion_conv_strides,
      projection_dim=self.motion_latent_dim,
    )

    mlp_output_dim = (
      self.distribution.input_dim if self.distribution is not None else output_dim
    )
    self.mlp = _build_mlp(
      input_dim=self._get_latent_dim(),
      output_dim=mlp_output_dim,
      hidden_dims=hidden_dims,
      activation=activation,
      layer_norm=self.layer_norm,
    )
    if self.distribution is not None:
      self.distribution.init_mlp_weights(self.mlp)

  def _get_latent_dim(self) -> int:
    remainder_dim = self.obs_dim - self.motion_obs_dim
    return remainder_dim + self.single_motion_obs_dim + self.motion_latent_dim

  def get_latent(
    self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
  ) -> torch.Tensor:
    obs_flat = super().get_latent(obs, masks, hidden_state)
    motion_obs = obs_flat[:, : self.motion_obs_dim]
    remainder = obs_flat[:, self.motion_obs_dim :]
    current_motion_obs = motion_obs[:, : self.single_motion_obs_dim]
    motion_latent = self.motion_encoder(motion_obs)
    return torch.cat((remainder, current_motion_obs, motion_latent), dim=-1)


class _OnnxYahmpActorModel(nn.Module):
  """ONNX wrapper for YahmpActorModel."""

  is_recurrent: bool = False

  def __init__(self, model: YahmpActorModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.history_encoder = copy.deepcopy(model.history_encoder)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim
    self.current_motion_obs_dim = model.current_motion_obs_dim
    self.current_obs_dim = model.current_obs_dim
    self.proprio_obs_dim = model.proprio_obs_dim
    self.history_obs_dim = model.history_obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    motion_end = self.current_motion_obs_dim
    current_end = self.current_obs_dim
    history_end = current_end + self.history_obs_dim

    motion_obs = x[:, :motion_end]
    proprio_obs = x[:, motion_end:current_end]
    history_obs = x[:, current_end:history_end]
    history_latent = self.history_encoder(history_obs)
    out = self.mlp(torch.cat((motion_obs, proprio_obs, history_latent), dim=-1))
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]


class _OnnxYahmpFutureActorModel(nn.Module):
  """ONNX wrapper for YahmpFutureActorModel."""

  is_recurrent: bool = False

  def __init__(self, model: YahmpFutureActorModel, verbose: bool = False) -> None:
    super().__init__()
    self.verbose = verbose
    self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
    self.motion_encoder = copy.deepcopy(model.motion_encoder)
    self.history_encoder = copy.deepcopy(model.history_encoder)
    self.mlp = copy.deepcopy(model.mlp)
    if model.distribution is not None:
      self.deterministic_output = model.distribution.as_deterministic_output_module()
    else:
      self.deterministic_output = nn.Identity()
    self.input_size = model.obs_dim
    self.motion_obs_dim = model.motion_obs_dim
    self.single_motion_obs_dim = model.single_motion_obs_dim
    self.proprio_obs_dim = model.proprio_obs_dim
    self.history_obs_dim = model.history_obs_dim

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.obs_normalizer(x)
    motion_end = self.motion_obs_dim
    proprio_end = motion_end + self.proprio_obs_dim
    history_end = proprio_end + self.history_obs_dim

    motion_obs = x[:, :motion_end]
    proprio_obs = x[:, motion_end:proprio_end]
    history_obs = x[:, proprio_end:history_end]
    current_motion_obs = motion_obs[:, : self.single_motion_obs_dim]
    motion_latent = self.motion_encoder(motion_obs)
    history_latent = self.history_encoder(history_obs)
    out = self.mlp(
      torch.cat(
        (current_motion_obs, proprio_obs, motion_latent, history_latent), dim=-1
      )
    )
    return self.deterministic_output(out)

  def get_dummy_inputs(self) -> tuple[torch.Tensor]:
    return (torch.zeros(1, self.input_size),)

  @property
  def input_names(self) -> list[str]:
    return ["obs"]

  @property
  def output_names(self) -> list[str]:
    return ["actions"]
