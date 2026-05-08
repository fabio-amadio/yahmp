from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import torch
from torch import nn
from vector_quantize_pytorch.vector_quantize_pytorch import VectorQuantize


@dataclass
class RVQCfg:
    dim: int
    num_quantizers: int = 8
    codebook_size: int = 1024
    codebook_dim: Optional[int] = None
    shared_codebook: bool = False

    quantize_dropout: bool = True

    # VectorQuantize kwargs
    decay: float = 0.99
    eps: float = 1e-5
    commitment_weight: float = 1.0
    kmeans_init: bool = False
    kmeans_iters: int = 10
    rotation_trick: bool = True


class ResidualVQ(nn.Module):
    """
    Residual Vector Quantization (RVQ) for y in (B, D).
    Quantizes sequentially, subtracting detached quantized vector each step.

    Returns:
      y_hat: (B, D)
      info: dict with indices, loss_vq, losses_per_layer, num_active
    """

    def __init__(
        self, cfg: RVQCfg, dropout_cutoff_index: int = 0, dropout_multiple_of: int = 1
    ):
        super().__init__()
        self.cfg = cfg
        dim = int(cfg.dim)
        codebook_dim = int(cfg.codebook_dim) if cfg.codebook_dim is not None else dim

        self.dim = dim
        self.codebook_dim = codebook_dim
        self.num_quantizers = int(cfg.num_quantizers)

        # optional projection if codebook_dim != dim
        self.project_in = (
            nn.Linear(dim, codebook_dim) if codebook_dim != dim else nn.Identity()
        )
        self.project_out = (
            nn.Linear(codebook_dim, dim) if codebook_dim != dim else nn.Identity()
        )

        self.quantize_dropout = bool(cfg.quantize_dropout) and self.num_quantizers > 1
        self.dropout_cutoff_index = dropout_cutoff_index
        self.dropout_multiple_of = dropout_multiple_of

        assert self.dropout_cutoff_index >= 0
        assert self.dropout_multiple_of >= 1

        layers = []
        for _ in range(self.num_quantizers):
            vq = VectorQuantize(
                dim=codebook_dim,
                codebook_size=int(cfg.codebook_size),
                decay=float(cfg.decay),
                eps=float(cfg.eps),
                commitment_weight=float(cfg.commitment_weight),
                kmeans_init=bool(cfg.kmeans_init),
                kmeans_iters=int(cfg.kmeans_iters),
                rotation_trick=bool(cfg.rotation_trick),
            )
            layers.append(vq)

        self.layers = nn.ModuleList(layers)

        # optional shared codebook across quantizers
        if cfg.shared_codebook:
            first = self.layers[0]
            shared = first._codebook
            for layer in self.layers[1:]:
                layer._codebook = shared

    def _round_up_multiple(self, num: int, mult: int) -> int:
        return ((num + mult - 1) // mult) * mult

    def _sample_num_active(self) -> int:
        """
        Sample M in [1..num_quantizers] (as index cutoff).
        Paper-style: choose a cutoff index and use quantizers 0..cutoff.
        """
        if (not self.training) or (not self.quantize_dropout):
            return self.num_quantizers

        # choose dropout cutoff index
        # active quantizers = cutoff_idx + 1
        cutoff_low = self.dropout_cutoff_index
        cutoff_high = self.num_quantizers - 1
        cutoff_idx = torch.randint(
            low=cutoff_low, high=cutoff_high + 1, size=(1,)
        ).item()

        if self.dropout_multiple_of != 1:
            # make (#active) a multiple of dropout_multiple_of
            active = self._round_up_multiple(cutoff_idx + 1, self.dropout_multiple_of)
            active = min(active, self.num_quantizers)
            return int(active)

        return int(cutoff_idx + 1)

    def forward(self, y: torch.Tensor) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        y: (B, D) residual to quantize
        """
        assert y.dim() == 2, f"Expected (B, D), got {tuple(y.shape)}"
        assert y.shape[-1] == self.dim, f"Expected dim={self.dim}, got {y.shape[-1]}"

        y_in = self.project_in(y)  # (B, codebook_dim)
        residual = y_in
        y_hat = torch.zeros_like(y_in)

        num_active = self._sample_num_active()

        all_indices = []
        all_losses = []

        for qi, layer in enumerate(self.layers):
            if qi >= num_active:
                break

            # VectorQuantize usually returns: quantized, indices, loss
            quantized, indices, loss = layer(residual)

            # residual quantization step (detach quantized so only current layer gets signal through residual path)
            residual = residual - quantized.detach()
            y_hat = y_hat + quantized

            all_indices.append(indices)  # (B,) typically
            all_losses.append(loss)  # scalar or (B,) depending on library version

        y_hat = self.project_out(y_hat)

        # stack indices to (B, num_active)
        indices = (
            torch.stack(all_indices, dim=-1)
            if len(all_indices) > 0
            else torch.empty((y.shape[0], 0), device=y.device, dtype=torch.long)
        )

        # losses to tensor
        # Make losses_per_layer shape (num_active,) or (B, num_active) depending on VQ implementation.
        # We'll reduce to scalar loss_vq robustly.
        # losses
        if len(all_losses) > 0:
            # each loss is scalar (0-d); make (num_active,)
            losses_per_layer = torch.stack([l.reshape(()) for l in all_losses], dim=0)
            loss_vq = losses_per_layer.mean()
        else:
            losses_per_layer = torch.zeros((0,), device=y.device, dtype=y.dtype)
            loss_vq = torch.zeros((), device=y.device, dtype=y.dtype)

        info = {
            "indices": indices,  # (B, num_active)
            "loss_vq": loss_vq,  # scalar
            "losses_per_layer": losses_per_layer,
            "num_active": torch.tensor(num_active, device=y.device),
        }
        return y_hat, info
