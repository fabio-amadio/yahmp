from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_nn_activation
from tensordict import TensorDict

from yahmp.rl.policy import MotionEncoder, _build_mlp
from yahmp.rl.residual_vq import ResidualVQ, RVQCfg


class _PriorNet(nn.Module):
    """Prior latent: zp = f_theta(s)."""

    def __init__(
        self,
        s_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        layer_norm: bool,
        distribution_cfg=None,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            input_dim=int(s_dim),
            output_dim=int(latent_dim),
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, s: torch.Tensor) -> torch.Tensor:
        return self.net(s)


class _PosteriorNet(nn.Module):
    """Posterior latent: z = f_phi(s, goal)."""

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
        latent_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.net = _build_mlp(
            input_dim=int(s_dim) + int(goal_dim),
            output_dim=int(latent_dim),
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((s, goal), dim=-1))


class _ActionDecoder(nn.Module):
    """Action decoder with state injection: a_hat = pi_low(s, z_hat)."""

    def __init__(
        self,
        s_dim: int,
        latent_dim: int,
        action_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
    ) -> None:
        super().__init__()
        self.activation_fn = resolve_nn_activation(activation)
        layers: list[nn.Module] = []
        in_size = int(latent_dim)
        for h in hidden_dims:
            layers.append(nn.Linear(in_size + int(s_dim), int(h)))
            in_size = int(h)
        self.layers = nn.ModuleList(layers)
        self.mu_head = nn.Linear(in_size, int(action_dim))

    def forward(self, s: torch.Tensor, z_hat: torch.Tensor) -> torch.Tensor:
        x = z_hat
        for layer in self.layers:
            x = self.activation_fn(layer(torch.cat((s, x), dim=-1)))
        return self.mu_head(x)


class YahmpImitationModel(MLPModel):
    """Imitation policy with prior, posterior, residual VQ, and action decoder.

    Observation layout matches `YahmpEncoderDecoderActorModel`:
      [current_motion | proprio | history]

    Forward flow (training):
      1. history_encoder(history) -> history_latent
      2. s = cat(proprio, history_latent),  goal = current_motion
      3. zp = prior(s)
      4. z  = posterior(s, goal)
      5. y  = z - zp.detach()
      6. y_hat, vq_info = RVQ(y)
      7. z_hat = zp.detach() + y_hat
      8. a_hat = action_decoder(s, z_hat)

    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        current_motion_obs_dim: int = 0,
        proprio_obs_dim: int = 0,
        history_steps: int = 10,
        distribution_cfg: dict | None = None,
        history_latent_dim: int = 128,
        history_conv_channels: tuple[int, ...] | list[int] = (64, 32),
        history_conv_kernel_sizes: tuple[int, ...] | list[int] = (4, 2),
        history_conv_strides: tuple[int, ...] | list[int] = (2, 1),
        latent_dim: int = 128,
        prior_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        posterior_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        layer_norm: bool = True,
        rvq_num_quantizers: int = 8,
        rvq_codebook_size: int = 1024,
        rvq_codebook_dim: int | None = None,
        rvq_shared_codebook: bool = False,
        rvq_quantize_dropout: bool = True,
        rvq_decay: float = 0.99,
        rvq_eps: float = 1e-5,
        rvq_commitment_weight: float = 1.0,
        rvq_kmeans_init: bool = False,
        rvq_kmeans_iters: int = 10,
        rvq_rotation_trick: bool = True,
    ) -> None:
        self.current_motion_obs_dim = int(current_motion_obs_dim)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.history_steps = int(history_steps)
        self.history_latent_dim = int(history_latent_dim)
        self.latent_dim = int(latent_dim)
        self.layer_norm = bool(layer_norm)
        self.action_dim = int(output_dim)

        if self.current_motion_obs_dim <= 0:
            raise ValueError(
                f"`current_motion_obs_dim` must be positive, got {self.current_motion_obs_dim}."
            )
        if self.proprio_obs_dim <= 0:
            raise ValueError(
                f"`proprio_obs_dim` must be positive, got {self.proprio_obs_dim}."
            )
        if self.history_steps <= 0:
            raise ValueError(
                f"`history_steps` must be positive, got {self.history_steps}."
            )
        if self.latent_dim <= 0:
            raise ValueError(f"`latent_dim` must be positive, got {self.latent_dim}.")

        self.current_obs_dim = self.current_motion_obs_dim + self.proprio_obs_dim
        self.history_obs_dim = self.proprio_obs_dim * self.history_steps

        # MLPModel always builds `self.mlp`. We don't use it (the action decoder owns
        # the output head). We pass `distribution_cfg=None` so `output_dim` is used
        # directly, and replace `self.mlp` with `nn.Identity` after super().__init__
        # to drop the unused parameters.
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=None,
        )
        self.mlp = nn.Identity()

        expected_obs_dim = self.current_obs_dim + self.history_obs_dim
        if self.obs_dim != expected_obs_dim:
            raise ValueError(
                "YahmpImitationModel observation dimension mismatch: "
                f"got {self.obs_dim}, expected {expected_obs_dim} "
                f"({self.current_obs_dim} current + {self.history_obs_dim} history)."
            )

        self.history_encoder = MotionEncoder(
            input_dim_per_step=self.proprio_obs_dim,
            num_steps=self.history_steps,
            activation=activation,
            conv_channels=history_conv_channels,
            conv_kernel_sizes=history_conv_kernel_sizes,
            conv_strides=history_conv_strides,
            projection_dim=self.history_latent_dim,
        )

        self.s_dim = self.proprio_obs_dim + self.history_latent_dim
        self.goal_dim = self.current_motion_obs_dim

        self.prior = _PriorNet(
            s_dim=self.s_dim,
            latent_dim=self.latent_dim,
            hidden_dims=prior_hidden_dims,
            activation=activation,
            layer_norm=self.layer_norm,
        )
        self.posterior = _PosteriorNet(
            s_dim=self.s_dim,
            goal_dim=self.goal_dim,
            latent_dim=self.latent_dim,
            hidden_dims=posterior_hidden_dims,
            activation=activation,
            layer_norm=self.layer_norm,
        )
        self.action_decoder = _ActionDecoder(
            s_dim=self.s_dim,
            latent_dim=self.latent_dim,
            action_dim=self.action_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )

        rvq_cfg = RVQCfg(
            dim=self.latent_dim,
            num_quantizers=int(rvq_num_quantizers),
            codebook_size=int(rvq_codebook_size),
            codebook_dim=None if rvq_codebook_dim is None else int(rvq_codebook_dim),
            shared_codebook=bool(rvq_shared_codebook),
            quantize_dropout=bool(rvq_quantize_dropout),
            decay=float(rvq_decay),
            eps=float(rvq_eps),
            commitment_weight=float(rvq_commitment_weight),
            kmeans_init=bool(rvq_kmeans_init),
            kmeans_iters=int(rvq_kmeans_iters),
            rotation_trick=bool(rvq_rotation_trick),
        )
        self.rvq = ResidualVQ(rvq_cfg)

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

    def _state_and_goal(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (s, goal) from a TensorDict obs by re-using MLPModel's normalizer."""
        obs_flat = super().get_latent(obs, masks, hidden_state)
        motion_obs, proprio_obs, history_obs = self._split_obs(obs_flat)
        history_latent = self.history_encoder(history_obs)
        s = torch.cat((proprio_obs, history_latent), dim=-1)
        goal = motion_obs
        return s, goal

    def forward(
        self,
        obs: TensorDict,
        masks: torch.Tensor | None = None,
        hidden_state=None,
        stochastic_output: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Full imitation forward pass — returns a dict of intermediate tensors."""
        del stochastic_output  # kept for MLPModel signature compatibility.
        s, goal = self._state_and_goal(obs, masks, hidden_state)

        zp = self.prior(s)
        z = self.posterior(s, goal)
        zp_sg = zp.detach()
        y = z - zp_sg
        y_hat, vq_info = self.rvq(y)
        z_hat = y_hat + zp_sg
        a_hat = self.action_decoder(s, z_hat)

        return {
            "a_hat": a_hat,
            "zp": zp,
            "z": z,
            "y": y,
            "y_hat": y_hat,
            "z_hat": z_hat,
            "vq_info": vq_info,
        }

    # @torch.no_grad()
    # def get_action(self, obs: TensorDict) -> torch.Tensor:
    #   """Deployment-time action: prior + decoder only."""
    #   s, goal = self._state_and_goal(obs)
    #   zp = self.prior(s)
    #   return self.action_decoder(s, zp)

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxYahmpImitationModel(self, verbose=verbose)


class _OnnxYahmpImitationModel(nn.Module):
    """ONNX export wrapper. Deployment uses the prior + decoder only."""

    is_recurrent: bool = False

    def __init__(self, model: YahmpImitationModel, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.prior = copy.deepcopy(model.prior)
        self.posterior = copy.deepcopy(model.posterior)
        self.rvq = copy.deepcopy(model.rvq)
        self.action_decoder = copy.deepcopy(model.action_decoder)

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

        proprio_obs = x[:, motion_end:current_end]
        history_obs = x[:, current_end:history_end]
        motion_obs = x[:, :motion_end]
        history_latent = self.history_encoder(history_obs)
        s = torch.cat((proprio_obs, history_latent), dim=-1)

        zp = self.prior(s)
        z = self.posterior(s, motion_obs)
        zp_sg = zp.detach()
        y = z - zp_sg
        y_hat, vq_info = self.rvq(y)
        z_hat = y_hat + zp_sg
        action = self.action_decoder(s, z_hat)
        return action

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
