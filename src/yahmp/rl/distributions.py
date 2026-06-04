"""Custom distributions for the YAHMP RL stack.

Provides ``MultiCategoricalDistribution``, used by ``YahmpLocomotionActorModel``
to sample tuples of RVQ codebook indices instead of continuous latent vectors.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from rsl_rl.modules.distribution import Distribution
from torch.distributions import Categorical


class MultiCategoricalDistribution(Distribution):
    """Stack of independent Categoricals, one per RVQ codebook layer.

    The actor's MLP head produces a flat logits tensor of shape
    ``(B, num_heads * codebook_size)``. ``update`` reshapes it to
    ``(B, num_heads, codebook_size)`` and instantiates a single
    ``torch.distributions.Categorical`` over the last dim.

    Conventions (to match ``rsl_rl``'s ``Distribution`` base):
      * ``output_dim`` = ``num_heads`` (the action is a tuple of ``num_heads`` ints)
      * ``input_dim``  = ``num_heads * codebook_size`` (flat logits)
      * ``sample()``, ``log_prob()``, ``params``, ``entropy`` and
        ``kl_divergence`` all aggregate (sum) over the heads axis.
    """

    def __init__(self, output_dim: int, codebook_size: int) -> None:
        super().__init__(output_dim)
        self.num_heads = int(output_dim)
        self.codebook_size = int(codebook_size)
        self._distribution: Categorical | None = None
        self._logits: torch.Tensor | None = None
        Categorical.set_default_validate_args(False)

    @property
    def input_dim(self) -> int:
        return self.num_heads * self.codebook_size

    def update(self, mlp_output: torch.Tensor) -> None:
        if mlp_output.dim() != 2 or mlp_output.shape[-1] != self.input_dim:
            raise ValueError(
                "MultiCategoricalDistribution.update expected (B, "
                f"{self.input_dim}) logits, got {tuple(mlp_output.shape)}."
            )
        logits = mlp_output.view(-1, self.num_heads, self.codebook_size)
        self._logits = logits
        self._distribution = Categorical(logits=logits)

    def sample(self) -> torch.Tensor:
        assert self._distribution is not None, "Call update() before sample()."
        return self._distribution.sample().to(torch.int64)

    def deterministic_output(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output.view(-1, self.num_heads, self.codebook_size)
        return logits.argmax(dim=-1).to(torch.int64)

    def as_deterministic_output_module(self) -> nn.Module:
        return _ArgmaxDeterministicOutput(self.num_heads, self.codebook_size)

    @property
    def mean(self) -> torch.Tensor:
        assert self._logits is not None, "Call update() before mean."
        return self._logits.argmax(dim=-1).to(torch.float32)

    @property
    def std(self) -> torch.Tensor:
        # Not a meaningful concept for a categorical; return zeros for API parity.
        assert self._logits is not None, "Call update() before std."
        return torch.zeros(
            self._logits.shape[0], self.num_heads, device=self._logits.device
        )

    @property
    def entropy(self) -> torch.Tensor:
        assert self._distribution is not None, "Call update() before entropy."
        return self._distribution.entropy().sum(dim=-1)

    @property
    def params(self) -> tuple[torch.Tensor, ...]:
        assert self._logits is not None, "Call update() before params."
        return (self._logits,)

    def log_prob(self, outputs: torch.Tensor) -> torch.Tensor:
        assert self._distribution is not None, "Call update() before log_prob."
        if outputs.dtype != torch.int64:
            outputs = outputs.to(torch.int64)
        return self._distribution.log_prob(outputs).sum(dim=-1)

    def kl_divergence(
        self,
        old_params: tuple[torch.Tensor, ...],
        new_params: tuple[torch.Tensor, ...],
    ) -> torch.Tensor:
        old_logits = old_params[0]
        new_logits = new_params[0]
        old_dist = Categorical(logits=old_logits)
        new_dist = Categorical(logits=new_logits)
        return torch.distributions.kl_divergence(old_dist, new_dist).sum(dim=-1)

    def init_mlp_weights(self, mlp: nn.Module) -> None:
        # No special initialization required; the actor overwrites mlp with Identity.
        return


class _ArgmaxDeterministicOutput(nn.Module):
    """Exportable module that maps flat logits to per-head argmax indices."""

    def __init__(self, num_heads: int, codebook_size: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.codebook_size = int(codebook_size)

    def forward(self, mlp_output: torch.Tensor) -> torch.Tensor:
        logits = mlp_output.view(-1, self.num_heads, self.codebook_size)
        return logits.argmax(dim=-1).to(torch.int64)
