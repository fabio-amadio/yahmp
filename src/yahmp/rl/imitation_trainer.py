from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ImitationLossWeights:
    action: float = 1.0
    mm: float = 0.1
    reg: float = 0.05
    vq: float = 1.0


@dataclass
class ImitationTrainerCfg:
    lr: float = 2e-4
    betas: tuple[float, float] = (0.9, 0.999)
    weight_decay: float = 0.0
    grad_clip_norm: float = 1.0
    mm_warmup_steps: int = 0
    mm_start: float = 0.1
    mm_end: float = 1.0


class ImitationTrainer:
    """Loss + optimizer wrapper for `YahmpImitationModel`.

    The trainer is environment-agnostic: it consumes batches with
    ``obs`` (a TensorDict) and ``a_expert`` (the action target produced
    by the frozen expert policy), and runs a single supervised step
    combining action reconstruction, margin minimization, optional
    temporal regularization, and the RVQ commitment loss.
    """

    def __init__(
        self,
        model: nn.Module,
        loss_weights: ImitationLossWeights,
        cfg: ImitationTrainerCfg,
        device: str | torch.device = "cuda",
    ) -> None:
        self.model = model
        self.loss_weights = loss_weights
        self.cfg = cfg

        self.device = torch.device(device)
        self.model.to(self.device)

        params = [p for p in self.model.parameters() if p.requires_grad]
        if len(params) == 0:
            raise RuntimeError(
                "ImitationTrainer received a model with no trainable parameters."
            )

        self.optim = torch.optim.Adam(
            params,
            lr=cfg.lr,
            betas=cfg.betas,
            weight_decay=cfg.weight_decay,
        )

        self.global_step = 0
        self._prev_out: dict[str, torch.Tensor] | None = None
        self._prev_size: int | None = None

    @staticmethod
    def _masked_mse(
        x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor
    ) -> torch.Tensor:
        if mask.dim() == 1:
            mask = (~mask).unsqueeze(-1)
        mask_f = mask.to(dtype=x.dtype)
        per = ((x - y) ** 2).mean(dim=-1, keepdim=True)
        denom = mask_f.sum().clamp_min(1.0)
        return (per * mask_f).sum() / denom

    def compute_imitation_losses(
        self,
        out: dict[str, torch.Tensor],
        a_expert: torch.Tensor,
        weights: ImitationLossWeights,
        prev_out: dict[str, torch.Tensor] | None = None,
        valid_prev: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        a_hat = out["a_hat"]
        zp = out["zp"]
        y_hat = out["y_hat"]
        z_hat = out["z_hat"]

        loss_action = F.mse_loss(a_hat, a_expert)
        loss_mm = F.mse_loss(z_hat, zp)

        loss_reg = torch.zeros((), device=a_hat.device, dtype=a_hat.dtype)
        if weights.reg > 0.0 and prev_out is not None and valid_prev is not None:
            # `valid_prev` is True where the previous step terminated — invert
            # so the mask selects same-episode transitions for temporal reg.
            same_ep = ~valid_prev
            loss_reg = self._masked_mse(
                zp, prev_out["zp"], same_ep
            ) + self._masked_mse(y_hat, prev_out["y_hat"], same_ep)

        loss_vq = torch.zeros((), device=a_hat.device, dtype=a_hat.dtype)
        vq_info = out.get("vq_info")
        if isinstance(vq_info, dict) and "loss_vq" in vq_info:
            loss_vq = vq_info["loss_vq"]

        loss_total = (
            weights.action * loss_action
            + weights.mm * loss_mm
            + weights.reg * loss_reg
            + weights.vq * loss_vq
        )

        return {
            "loss_total": loss_total,
            "loss_action": loss_action,
            "loss_mm": loss_mm,
            "loss_reg": loss_reg,
            "loss_commit": loss_vq,
        }

    def train_step(self, batch: dict[str, Any]) -> dict[str, float]:
        self.model.train()

        obs = batch["obs"]
        a_expert = batch["a_expert"].to(self.device)
        valid_prev = batch.get("valid_prev")
        if valid_prev is not None:
            valid_prev = valid_prev.to(self.device, dtype=torch.bool)

        self.optim.zero_grad(set_to_none=True)

        out = self.model(obs)

        batch_size = out["zp"].shape[0]
        if self._prev_out is None or self._prev_size != batch_size:
            self._prev_out = {
                "zp": out["zp"].detach().clone(),
                "y_hat": out["y_hat"].detach().clone(),
            }
            self._prev_size = batch_size

        if self.cfg.mm_warmup_steps > 0:
            t = min(1.0, self.global_step / float(self.cfg.mm_warmup_steps))
            self.loss_weights.mm = (1.0 - t) * self.cfg.mm_start + t * self.cfg.mm_end
        else:
            self.loss_weights.mm = self.cfg.mm_end

        losses = self.compute_imitation_losses(
            out=out,
            a_expert=a_expert,
            weights=self.loss_weights,
            prev_out=self._prev_out,
            valid_prev=valid_prev,
        )
        loss_total = losses["loss_total"]

        self._prev_out["zp"] = out["zp"].detach().clone()
        self._prev_out["y_hat"] = out["y_hat"].detach().clone()

        loss_total.backward()

        if self.cfg.grad_clip_norm and self.cfg.grad_clip_norm > 0:
            grad_norm = torch.nn.utils.clip_grad_norm_(
                [p for p in self.model.parameters() if p.requires_grad],
                self.cfg.grad_clip_norm,
            )
            grad_norm_val = float(grad_norm.detach().item())
        else:
            grad_norm_val = 0.0

        self.optim.step()
        self.global_step += 1

        stats = {k: float(v.detach().item()) for k, v in losses.items()}
        stats["optim/grad_norm"] = grad_norm_val
        stats["optim/lr"] = float(self.optim.param_groups[0]["lr"])
        stats["weights/mm"] = float(self.loss_weights.mm)

        vq_info = out.get("vq_info")
        if isinstance(vq_info, dict):
            if "num_active" in vq_info:
                stats["rvq/num_active"] = float(vq_info["num_active"].detach().item())
            if "loss_vq" in vq_info:
                stats["rvq/loss_vq"] = float(vq_info["loss_vq"].detach().item())

        return stats

    def state_dict(self) -> dict[str, Any]:
        return {
            "model": self.model.state_dict(),
            "optim": self.optim.state_dict(),
            "global_step": int(self.global_step),
        }

    def load_state_dict(self, sd: dict[str, Any], strict: bool = True) -> None:
        self.model.load_state_dict(sd["model"], strict=strict)
        self.optim.load_state_dict(sd["optim"])
        self.global_step = int(sd.get("global_step", 0))
