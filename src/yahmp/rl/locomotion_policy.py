"""Hierarchical actor for the YAHMP omnidirectional locomotion task.

The actor reuses the imitation pipeline (prior + Residual VQ + action decoder)
as a frozen low-level controller and trains a new high-level encoder that maps
``(state, velocity_command) -> latent z`` on top of it.

Forward pipeline:

    1. obs_normalizer(obs) -> obs_flat
    2. split obs_flat into ``[g_task | proprio | history_obs]``
    3. history_encoder(history_obs) -> history_latent          [TRAINABLE]
    4. s_rich = cat(proprio, history_latent)
    5. z = high_level(cat(s_rich, g_task))                     [TRAINABLE]
    6. zp = prior(s_rich).detach()                             [FROZEN]
    7. y_hat, _ = rvq(z - zp)                                  [FROZEN]
    8. z_hat = zp + y_hat
    9. a_mean = action_decoder(s_rich, z_hat)                  [FROZEN]

PPO samples actions ``a ~ N(a_mean, σ)`` with a learnable diagonal std.
Gradients flow back through the frozen modules to update the high-level net,
history_encoder and the action std parameter; the frozen submodules'
weights themselves do not move (``requires_grad=False`` + permanent eval mode).
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from yahmp.rl.imitation_RVQ_policy import _ActionDecoder, _PosteriorNet, _PriorNet
from yahmp.rl.policy import MotionEncoder
from yahmp.rl.residual_vq import ResidualVQ, RVQCfg


class YahmpLocomotionActorModel(MLPModel):
    """Actor for the YAHMP locomotion task with a frozen imitation backbone."""

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
        task_goal_obs_dim: int = 0,
        proprio_obs_dim: int = 0,
        history_steps: int = 10,
        history_latent_dim: int = 128,
        history_conv_channels: tuple[int, ...] | list[int] = (64, 32),
        history_conv_kernel_sizes: tuple[int, ...] | list[int] = (4, 2),
        history_conv_strides: tuple[int, ...] | list[int] = (2, 1),
        latent_dim: int = 128,
        high_level_hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        layer_norm: bool = True,
        rvq_num_quantizers: int = 8,
        rvq_codebook_size: int = 1024,
        rvq_codebook_dim: int | None = None,
        rvq_shared_codebook: bool = False,
        rvq_decay: float = 0.99,
        rvq_eps: float = 1e-5,
        rvq_commitment_weight: float = 1.0,
        rvq_rotation_trick: bool = True,
    ) -> None:
        self.task_goal_obs_dim = int(task_goal_obs_dim)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.history_steps = int(history_steps)
        self.history_latent_dim = int(history_latent_dim)
        self.latent_dim = int(latent_dim)
        self.layer_norm = bool(layer_norm)
        self.action_dim = int(output_dim)

        if self.task_goal_obs_dim <= 0:
            raise ValueError(
                f"`task_goal_obs_dim` must be positive, got {self.task_goal_obs_dim}."
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

        self.current_obs_dim = self.task_goal_obs_dim + self.proprio_obs_dim
        self.history_obs_dim = self.proprio_obs_dim * self.history_steps

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
        self.mlp = nn.Identity()

        expected_obs_dim = self.current_obs_dim + self.history_obs_dim
        if self.obs_dim != expected_obs_dim:
            raise ValueError(
                "YahmpLocomotionActorModel observation dimension mismatch: "
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
        self.goal_dim = self.task_goal_obs_dim

        self.prior = _PriorNet(
            s_dim=self.s_dim,
            latent_dim=self.latent_dim,
            hidden_dims=high_level_hidden_dims,
            activation=activation,
            layer_norm=self.layer_norm,
        )
        self.high_level = _PosteriorNet(
            s_dim=self.s_dim,
            goal_dim=self.goal_dim,
            latent_dim=self.latent_dim,
            hidden_dims=high_level_hidden_dims,
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
            quantize_dropout=False,
            decay=float(rvq_decay),
            eps=float(rvq_eps),
            commitment_weight=float(rvq_commitment_weight),
            kmeans_init=False,
            kmeans_iters=0,
            rotation_trick=bool(rvq_rotation_trick),
        )
        self.rvq = ResidualVQ(rvq_cfg)

        # ``prior`` and ``action_decoder`` are pure-MLP modules: freeze with
        # ``eval()`` + ``requires_grad=False``.
        # ``rvq`` is special — the underlying VectorQuantize layers only apply
        # the Straight-Through Estimator when ``self.training=True``, so the
        # module must stay in train mode to let gradients reach ``high_level``.
        # Codebook EMA updates are suppressed by setting ``freeze_codebook=True``
        # on each layer (see ``_freeze_rvq``).
        self._frozen_eval_submodules: tuple[str, ...] = ("prior", "action_decoder")

    def _get_latent_dim(self) -> int:
        # Not used: `self.mlp` is overwritten with `nn.Identity()` after init.
        return self.action_dim

    def _split_obs(
        self, obs_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        goal_end = self.task_goal_obs_dim
        current_end = self.current_obs_dim
        history_end = current_end + self.history_obs_dim
        return (
            obs_flat[:, :goal_end],
            obs_flat[:, goal_end:current_end],
            obs_flat[:, current_end:history_end],
        )

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
    ) -> torch.Tensor:
        obs_flat = super().get_latent(obs, masks, hidden_state)
        g_task, proprio, history_obs = self._split_obs(obs_flat)
        history_latent = self.history_encoder(history_obs)
        s_rich = torch.cat((proprio, history_latent), dim=-1)

        z = self.high_level(s_rich, g_task)
        zp = self.prior(s_rich).detach()
        y = z - zp
        y_hat, _ = self.rvq(y)
        z_hat = zp + y_hat
        return self.action_decoder(s_rich, z_hat)

    def train(self, mode: bool = True) -> "YahmpLocomotionActorModel":
        super().train(mode)
        for name in self._frozen_eval_submodules:
            module = getattr(self, name, None)
            if module is not None:
                module.eval()
        # Keep RVQ in train mode (for STE) but with codebook frozen.
        self.rvq.train(mode)
        return self

    def load_imitation_weights(
        self,
        imitation_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        copy_normalizer_proprio_history: bool = True,
    ) -> None:
        """Load frozen submodules (prior, rvq, action_decoder) from an imitation
        checkpoint and freeze their parameters.

        Optionally copies the proprio + history slots of the imitation model's
        ``obs_normalizer`` running statistics into this model's normalizer.
        The task-goal slot keeps fresh (zero/one) statistics.

        Args:
          imitation_state_dict: state_dict of a trained ``YahmpImitationModel``.
          strict: forward to ``Module.load_state_dict`` when loading each submodule.
          copy_normalizer_proprio_history: if True, slice the imitation normalizer
            and copy proprio + history portions into this model's normalizer.
        """
        by_prefix: dict[str, dict[str, torch.Tensor]] = {}
        for key, tensor in imitation_state_dict.items():
            prefix, _, sub_key = key.partition(".")
            if not sub_key:
                continue
            by_prefix.setdefault(prefix, {})[sub_key] = tensor

        missing: list[str] = []

        def _load(submodule_name: str) -> None:
            module = getattr(self, submodule_name)
            sd = by_prefix.get(submodule_name)
            if sd is None:
                missing.append(submodule_name)
                return
            module.load_state_dict(sd, strict=strict)

        for name in ("prior", "rvq", "action_decoder"):
            _load(name)

        if missing:
            raise KeyError(
                "YahmpLocomotionActorModel: missing submodules in imitation "
                f"checkpoint: {missing}. Expected keys with prefixes "
                "'prior.', 'rvq.', 'action_decoder.'."
            )

        self._freeze_frozen_submodules()

        if copy_normalizer_proprio_history and self.obs_normalization:
            self._copy_normalizer_proprio_history(imitation_state_dict)

    def _freeze_frozen_submodules(self) -> None:
        for name in self._frozen_eval_submodules:
            module = getattr(self, name)
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        self._freeze_rvq()

    def _freeze_rvq(self) -> None:
        """Freeze RVQ parameters and codebooks while keeping STE gradient flow.

        ``vector_quantize_pytorch`` applies the Straight-Through Estimator
        only when the VQ module is in train mode. We want gradient to flow
        through the quantizer (so the high-level net can learn), but we do
        *not* want EMA codebook updates. Setting ``freeze_codebook=True`` on
        each VectorQuantize layer disables the EMA updates; combined with
        ``requires_grad=False`` on parameters, the RVQ becomes a frozen but
        gradient-transparent module.
        """
        for p in self.rvq.parameters():
            p.requires_grad = False
        for layer in self.rvq.layers:
            layer.freeze_codebook = True
        self.rvq.train()

    def _copy_normalizer_proprio_history(
        self, imitation_state_dict: dict[str, torch.Tensor]
    ) -> None:
        """Copy the proprio slot of the imitation normalizer into this one.

        Imitation current-obs layout was ``[motion_ref | proprio]``; locomotion
        is ``[g_task | proprio]``. The per-step ``proprio`` block is the only
        section with identical semantics across the two pipelines, so only that
        slot is transferred. History slots (different per-step content) and the
        ``g_task`` slot are left at the (mean=0, var=1) init.

        The imitation ``count`` is also transferred to lock these stats in: PPO
        keeps calling ``obs_normalizer.update()`` each env step, and with
        ``count=0`` ``EmpiricalNormalization`` uses ``rate = batch_size / count``
        which collapses to ``1.0`` on the first call and overwrites the copied
        stats entirely. Restoring a large ``count`` makes the subsequent rate
        negligible, so the proprio stats stay consistent with imitation
        throughout PPO training.
        """
        imitation_mean = imitation_state_dict.get("obs_normalizer._mean")
        imitation_var = imitation_state_dict.get("obs_normalizer._var")
        imitation_std = imitation_state_dict.get("obs_normalizer._std")
        imitation_count = imitation_state_dict.get("obs_normalizer.count")
        if imitation_mean is None or imitation_var is None or imitation_std is None:
            return

        # Imitation obs layout: [imit_command | proprio | history(proprio * H)].
        # Recover the imitation command-block size from the normalizer total length,
        # since the history conv no longer encodes the motion ref per step.
        imit_total = int(imitation_mean.shape[-1])
        imit_motion_obs_dim = imit_total - self.proprio_obs_dim * (
            1 + self.history_steps
        )
        if imit_motion_obs_dim < 0:
            return

        proprio_lo_imit = imit_motion_obs_dim
        proprio_hi_imit = imit_motion_obs_dim + self.proprio_obs_dim
        proprio_lo_loc = self.task_goal_obs_dim
        proprio_hi_loc = self.task_goal_obs_dim + self.proprio_obs_dim

        with torch.no_grad():
            self.obs_normalizer._mean[:, proprio_lo_loc:proprio_hi_loc] = (  # type: ignore[attr-defined]
                imitation_mean[:, proprio_lo_imit:proprio_hi_imit]
            )
            self.obs_normalizer._var[:, proprio_lo_loc:proprio_hi_loc] = (  # type: ignore[attr-defined]
                imitation_var[:, proprio_lo_imit:proprio_hi_imit]
            )
            self.obs_normalizer._std[:, proprio_lo_loc:proprio_hi_loc] = (  # type: ignore[attr-defined]
                imitation_std[:, proprio_lo_imit:proprio_hi_imit]
            )
            if imitation_count is not None:
                self.obs_normalizer.count.copy_(imitation_count)  # type: ignore[attr-defined]

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxYahmpLocomotionActorModel(self, verbose=verbose)


class _OnnxYahmpLocomotionActorModel(nn.Module):
    """ONNX wrapper for ``YahmpLocomotionActorModel`` — deterministic mean action."""

    is_recurrent: bool = False

    def __init__(self, model: YahmpLocomotionActorModel, verbose: bool = False) -> None:
        super().__init__()
        self.verbose = verbose
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.history_encoder = copy.deepcopy(model.history_encoder)
        self.high_level = copy.deepcopy(model.high_level)
        self.prior = copy.deepcopy(model.prior)
        self.rvq = copy.deepcopy(model.rvq)
        self.action_decoder = copy.deepcopy(model.action_decoder)

        self.input_size = model.obs_dim
        self.task_goal_obs_dim = model.task_goal_obs_dim
        self.current_obs_dim = model.current_obs_dim
        self.proprio_obs_dim = model.proprio_obs_dim
        self.history_obs_dim = model.history_obs_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.obs_normalizer(x)
        goal_end = self.task_goal_obs_dim
        current_end = self.current_obs_dim
        history_end = current_end + self.history_obs_dim
        g_task = x[:, :goal_end]
        proprio = x[:, goal_end:current_end]
        history_obs = x[:, current_end:history_end]
        history_latent = self.history_encoder(history_obs)
        s_rich = torch.cat((proprio, history_latent), dim=-1)
        z = self.high_level(s_rich, g_task)
        zp = self.prior(s_rich)
        y = z - zp
        y_hat, _ = self.rvq(y)
        z_hat = zp + y_hat
        return self.action_decoder(s_rich, z_hat)

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
