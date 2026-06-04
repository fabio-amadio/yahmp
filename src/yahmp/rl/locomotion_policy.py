"""Hierarchical actor for the YAHMP omnidirectional locomotion task.

The actor reuses the imitation pipeline (history encoder + prior + Residual VQ
+ action decoder) as a frozen low-level controller and trains a new high-level
encoder that maps ``(state, velocity_command) -> categorical RVQ-indices`` on
top of it.

Forward pipeline (rollout / inference):

    1. obs_normalizer(obs) -> obs_flat
       (g_task slot is permanently pinned to identity — see P1 below)
    2. split obs_flat into ``[g_task | proprio | history_obs]``
    3. history_encoder(history_obs) -> history_latent        [FROZEN]
    4. s_rich = cat(proprio, history_latent)
    5. logits = high_level(s_rich, g_task)                   [TRAINABLE]
       logits shape: (B, num_heads * codebook_size) — flat
    6. Categorical sampling (PPO act): indices (B, num_heads) int64
       → categorical PPO stores indices in the rollout buffer.
    7. y_hat = sum_l codebook_l[indices[:, l]]               [FROZEN]
    8. zp = prior(s_rich)                                    [FROZEN]
    9. z_hat = zp + y_hat
   10. a = action_decoder(s_rich, z_hat)                     [FROZEN]
       continuous joint-position action passed to env.step.

Design rationale (P1+P2+P3):
  * **P1** — feed ``g_task`` *raw* (un-normalized) to the high-level: the
    EmpiricalNormalization for that slot is pinned to identity so the velocity
    command is preserved in physical units (~ ±1 m/s). This avoids the SNR
    imbalance between command (~±0.5) and joint_vel (~±15) when both share a
    fitted normalizer.
  * **P2** — the proprio + history slots of the locomotion normalizer are
    *warm-started* from an external checkpoint (expert or imitation). The
    EmpiricalNormalization ``count`` is also copied so subsequent PPO updates
    barely move the stats, keeping them consistent with the frozen backbone.
  * **P3** — the high-level produces a *Multi-Categorical* distribution over
    the RVQ codebook indices instead of a continuous latent that's then
    snapped to nearest codes. This decouples PPO gradients from the
    discrete bottleneck and matches the behaviour of the older
    g1_hybrid_prior repo where this design was empirically validated.
"""

from __future__ import annotations

import copy

import torch
import torch.nn as nn
from rsl_rl.models import MLPModel
from tensordict import TensorDict

from yahmp.rl.distributions import MultiCategoricalDistribution
from yahmp.rl.imitation_RVQ_policy import _ActionDecoder, _PriorNet
from yahmp.rl.policy import MotionEncoder, _build_mlp
from yahmp.rl.residual_vq import ResidualVQ, RVQCfg


class _CategoricalHighLevel(nn.Module):
    """Maps ``(s_rich, g_task)`` to flat per-head categorical logits.

    The output is a single tensor of shape
    ``(B, num_heads * codebook_size)``; the distribution module reshapes
    it to ``(B, num_heads, codebook_size)`` before instantiating a stack
    of independent Categoricals.
    """

    def __init__(
        self,
        s_dim: int,
        goal_dim: int,
        num_heads: int,
        codebook_size: int,
        hidden_dims: tuple[int, ...] | list[int],
        activation: str,
        layer_norm: bool,
    ) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.codebook_size = int(codebook_size)
        self.net = _build_mlp(
            input_dim=int(s_dim) + int(goal_dim),
            output_dim=self.num_heads * self.codebook_size,
            hidden_dims=hidden_dims,
            activation=activation,
            layer_norm=layer_norm,
        )

    def forward(self, s: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat((s, goal), dim=-1))


class YahmpLocomotionActorModel(MLPModel):
    """Hierarchical actor with frozen imitation backbone + categorical high-level."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = True,
        # Accepted for API parity with rl_cfg; not used (we attach our own
        # MultiCategoricalDistribution sized from ``rvq_num_quantizers`` and
        # ``rvq_codebook_size``).
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
        del distribution_cfg  # always replaced below.

        self.task_goal_obs_dim = int(task_goal_obs_dim)
        self.proprio_obs_dim = int(proprio_obs_dim)
        self.history_steps = int(history_steps)
        self.history_latent_dim = int(history_latent_dim)
        self.latent_dim = int(latent_dim)
        self.layer_norm = bool(layer_norm)
        self.action_dim = int(output_dim)
        self.num_active_codebooks = int(rvq_num_quantizers)
        self.codebook_size = int(rvq_codebook_size)

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
        if self.num_active_codebooks <= 0:
            raise ValueError(
                f"`rvq_num_quantizers` must be positive, got {self.num_active_codebooks}."
            )
        if self.codebook_size <= 0:
            raise ValueError(
                f"`rvq_codebook_size` must be positive, got {self.codebook_size}."
            )

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
            distribution_cfg=None,  # categorical distribution attached below.
        )
        # MLP head is unused — get_latent already returns the distribution input.
        self.mlp = nn.Identity()

        # Attach the categorical distribution AFTER super().__init__ so we can
        # size it from rvq_* explicitly (instead of inheriting output_dim=29).
        self.distribution = MultiCategoricalDistribution(
            output_dim=self.num_active_codebooks,
            codebook_size=self.codebook_size,
        )

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
        # Trainable: produces categorical logits over RVQ codebook indices.
        self.high_level = _CategoricalHighLevel(
            s_dim=self.s_dim,
            goal_dim=self.goal_dim,
            num_heads=self.num_active_codebooks,
            codebook_size=self.codebook_size,
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

        # ``history_encoder``, ``prior`` and ``action_decoder`` are deterministic
        # imitation-backbone modules: freeze with ``eval()`` +
        # ``requires_grad=False``. ``rvq`` is special: STE only fires in train
        # mode, and we want gradients through the codebook lookup at inference
        # time too, so we keep it in train mode but mark ``freeze_codebook``.
        self._frozen_eval_submodules: tuple[str, ...] = (
            "history_encoder",
            "prior",
            "action_decoder",
        )

        # P1: pin g_task slot of the normalizer to identity from the start.
        self._pin_gtask_normalizer_identity()

    def _get_latent_dim(self) -> int:
        # MLP is overwritten with nn.Identity right after super().__init__,
        # so this value only affects the size of the discarded MLP layer.
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

    def _build_s_rich(
        self, obs_flat: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(s_rich, g_task)`` given a normalized obs vector."""
        g_task, proprio, history_obs = self._split_obs(obs_flat)
        history_latent = self.history_encoder(history_obs)
        s_rich = torch.cat((proprio, history_latent), dim=-1)
        return s_rich, g_task

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None
    ) -> torch.Tensor:
        """Return flat categorical logits ``(B, num_heads * codebook_size)``."""
        obs_flat = super().get_latent(obs, masks, hidden_state)
        s_rich, g_task = self._build_s_rich(obs_flat)
        return self.high_level(s_rich, g_task)

    def lookup_codebook(self, indices: torch.Tensor) -> torch.Tensor:
        """Sum codeword vectors selected by ``indices`` across RVQ layers.

        Args:
          indices: (B, num_active_codebooks) int64.

        Returns:
          y_hat: (B, latent_dim) — the same quantity the RVQ forward path would
          produce given those indices, sans any STE / commitment loss.
        """
        if indices.dim() != 2 or indices.shape[-1] != self.num_active_codebooks:
            raise ValueError(
                "lookup_codebook expected indices of shape "
                f"(B, {self.num_active_codebooks}), got {tuple(indices.shape)}."
            )
        B = indices.shape[0]
        codebook_dim = self.rvq.codebook_dim
        y_hat = torch.zeros(B, codebook_dim, device=indices.device, dtype=torch.float32)
        for qi, layer in enumerate(self.rvq.layers[: self.num_active_codebooks]):
            embed = layer._codebook.embed  # (1, codebook_size, codebook_dim)
            codebook = embed[0]
            y_hat = y_hat + codebook[indices[:, qi]]
        return self.rvq.project_out(y_hat)

    def indices_to_continuous_action(
        self, obs: TensorDict, indices: torch.Tensor
    ) -> torch.Tensor:
        """Decode RVQ indices into the continuous joint-position action."""
        obs_flat = MLPModel.get_latent(self, obs, None, None)
        s_rich, _ = self._build_s_rich(obs_flat)
        zp = self.prior(s_rich)
        y_hat = self.lookup_codebook(indices.to(obs_flat.device))
        z_hat = zp + y_hat
        return self.action_decoder(s_rich, z_hat)

    def train(self, mode: bool = True) -> "YahmpLocomotionActorModel":
        super().train(mode)
        for name in self._frozen_eval_submodules:
            module = getattr(self, name, None)
            if module is not None:
                module.eval()
        # Keep RVQ in train mode for STE; codebook frozen via _freeze_rvq.
        self.rvq.train(mode)
        return self

    def load_imitation_weights(
        self,
        imitation_state_dict: dict[str, torch.Tensor],
        strict: bool = True,
        copy_normalizer_proprio_history: bool = True,
        expert_state_dict: dict[str, torch.Tensor] | None = None,
    ) -> None:
        """Load imitation submodules, freeze the backbone, set normalizer stats.

        Args:
          imitation_state_dict: state_dict of a trained ``YahmpImitationModel``.
            Used to copy ``history_encoder``, ``prior``, ``rvq`` and
            ``action_decoder``.
          strict: forwarded to ``Module.load_state_dict`` for each submodule.
          copy_normalizer_proprio_history: if True, source the proprio+history
            slots of this model's obs-normalizer from a checkpoint. The g_task
            slot is always pinned to identity (P1) regardless.
          expert_state_dict: optional EncDec expert checkpoint. When provided,
            its normalizer is used as the source for the proprio+history slots
            (P2). Otherwise the imitation checkpoint is used.
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

        for name in ("history_encoder", "prior", "rvq", "action_decoder"):
            _load(name)

        if missing:
            raise KeyError(
                "YahmpLocomotionActorModel: missing submodules in imitation "
                f"checkpoint: {missing}. Expected keys with prefixes "
                "'history_encoder.', 'prior.', 'rvq.', 'action_decoder.'."
            )

        self._freeze_frozen_submodules()

        if copy_normalizer_proprio_history and self.obs_normalization:
            if expert_state_dict is not None:
                self._copy_normalizer_proprio_history(
                    expert_state_dict, source_name="expert"
                )
            else:
                self._copy_normalizer_proprio_history(
                    imitation_state_dict, source_name="imitation"
                )
            self._pin_gtask_normalizer_identity()

    def _freeze_frozen_submodules(self) -> None:
        for name in self._frozen_eval_submodules:
            module = getattr(self, name)
            module.eval()
            for p in module.parameters():
                p.requires_grad = False
        self._freeze_rvq()

    def _freeze_rvq(self) -> None:
        """Freeze RVQ parameters + codebooks while keeping STE gradient flow."""
        for p in self.rvq.parameters():
            p.requires_grad = False
        for layer in self.rvq.layers:
            layer.freeze_codebook = True
        self.rvq.train()

    def _pin_gtask_normalizer_identity(self) -> None:
        """Force the g_task slot of the obs-normalizer to (mean=0, var=1, std=1).

        Called at construction and after every external normalizer load. The
        EmpiricalNormalization runs over the full obs vector, so subsequent
        ``update`` calls will still nudge the g_task slot — but with a large
        ``count`` (set by ``_copy_normalizer_proprio_history``) those nudges are
        negligibly small over a full training run.
        """
        if not self.obs_normalization:
            return
        end = self.task_goal_obs_dim
        if end <= 0:
            return
        with torch.no_grad():
            self.obs_normalizer._mean[:, :end].zero_()  # type: ignore[attr-defined]
            self.obs_normalizer._var[:, :end].fill_(1.0)  # type: ignore[attr-defined]
            self.obs_normalizer._std[:, :end].fill_(1.0)  # type: ignore[attr-defined]

    def _copy_normalizer_proprio_history(
        self,
        source_state_dict: dict[str, torch.Tensor],
        source_name: str = "source",
    ) -> None:
        """Copy proprio+history obs-normalizer slots from an external state_dict.

        The source can be either an imitation or expert checkpoint; both share
        the layout ``[motion-or-command(M) | proprio(P) | history(P*H)]``.
        Only the ``[proprio | history]`` tail is copied into the locomotion
        normalizer (which has layout ``[g_task(G) | proprio(P) | history(P*H)]``).

        The ``count`` buffer is also copied so subsequent in-place
        ``EmpiricalNormalization.update`` calls during PPO barely move the
        stats (rate ≈ batch_size / count ≈ 10⁻⁸).
        """
        source_mean = source_state_dict.get("obs_normalizer._mean")
        source_var = source_state_dict.get("obs_normalizer._var")
        source_std = source_state_dict.get("obs_normalizer._std")
        source_count = source_state_dict.get("obs_normalizer.count")
        if source_mean is None or source_var is None or source_std is None:
            print(
                f"[YahmpLocomotionActorModel] {source_name} state_dict missing "
                "obs_normalizer buffers — skipping normalizer copy."
            )
            return

        src_total = int(source_mean.shape[-1])
        tail_dim = self.proprio_obs_dim * (1 + self.history_steps)
        src_motion_obs_dim = src_total - tail_dim
        if src_motion_obs_dim < 0:
            print(
                f"[YahmpLocomotionActorModel] {source_name} normalizer too small "
                f"({src_total}) for tail dim {tail_dim}; skipping."
            )
            return

        tail_lo_src = src_motion_obs_dim
        tail_hi_src = src_motion_obs_dim + tail_dim
        tail_lo_loc = self.task_goal_obs_dim
        tail_hi_loc = self.task_goal_obs_dim + tail_dim
        loc_total = int(self.obs_normalizer._mean.shape[-1])  # type: ignore[attr-defined]
        if tail_hi_loc > loc_total:
            print(
                f"[YahmpLocomotionActorModel] locomotion normalizer too small "
                f"({loc_total}) for tail dim {tail_dim}; skipping."
            )
            return

        with torch.no_grad():
            self.obs_normalizer._mean[:, tail_lo_loc:tail_hi_loc] = (  # type: ignore[attr-defined]
                source_mean[:, tail_lo_src:tail_hi_src]
            )
            self.obs_normalizer._var[:, tail_lo_loc:tail_hi_loc] = (  # type: ignore[attr-defined]
                source_var[:, tail_lo_src:tail_hi_src]
            )
            self.obs_normalizer._std[:, tail_lo_loc:tail_hi_loc] = (  # type: ignore[attr-defined]
                source_std[:, tail_lo_src:tail_hi_src]
            )
            if source_count is not None:
                self.obs_normalizer.count.copy_(source_count)  # type: ignore[attr-defined]

        count_val = (
            int(self.obs_normalizer.count.item())  # type: ignore[attr-defined]
            if source_count is not None
            else 0
        )
        print(
            f"[YahmpLocomotionActorModel] Copied proprio+history obs-normalizer "
            f"from {source_name} (tail dims {tail_lo_src}:{tail_hi_src} → "
            f"{tail_lo_loc}:{tail_hi_loc}, count={count_val})."
        )

    def as_onnx(self, verbose: bool = False) -> nn.Module:
        return _OnnxYahmpLocomotionActorModel(self, verbose=verbose)


class _OnnxYahmpLocomotionActorModel(nn.Module):
    """ONNX wrapper: deterministic (argmax) categorical → continuous action."""

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
        self.num_heads = model.num_active_codebooks
        self.codebook_size = model.codebook_size

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
        logits = self.high_level(s_rich, g_task)
        logits = logits.view(-1, self.num_heads, self.codebook_size)
        indices = logits.argmax(dim=-1)  # (B, num_heads)
        # Codebook lookup + decode.
        B = indices.shape[0]
        codebook_dim = self.rvq.codebook_dim
        y_hat = torch.zeros(B, codebook_dim, device=indices.device, dtype=torch.float32)
        for qi, layer in enumerate(self.rvq.layers[: self.num_heads]):
            embed = layer._codebook.embed
            codebook = embed[0]
            y_hat = y_hat + codebook[indices[:, qi]]
        y_hat = self.rvq.project_out(y_hat)
        zp = self.prior(s_rich)
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
