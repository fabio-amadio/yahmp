"""PPO variant for the categorical hierarchical actor.

``YahmpCategoricalPPO`` differs from the stock ``rsl_rl.algorithms.PPO`` in two
places:

1. The rollout storage is allocated with ``actions_shape =
   [actor.num_active_codebooks]`` (one int per codebook layer) instead of
   ``[env.num_actions]`` (continuous joint dim), because the policy stores
   sampled RVQ indices in ``transition.actions``.
2. ``act`` samples the indices (via the base PPO logic, which already calls
   ``actor(..., stochastic_output=True)``), stores them in the rollout, then
   converts the indices to a continuous joint-position action via
   ``actor.indices_to_continuous_action`` so the env can be stepped.
"""

from __future__ import annotations

import torch
from rsl_rl.algorithms.ppo import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.models import MLPModel
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict


class YahmpCategoricalPPO(PPO):
    """PPO with categorical-indices storage and continuous env actions."""

    def act(self, obs: TensorDict) -> torch.Tensor:
        """Sample RVQ indices, store them in the transition, step env with continuous."""
        indices = super().act(obs)  # populates self.transition with indices
        actor = self.actor
        assert hasattr(actor, "indices_to_continuous_action"), (
            "YahmpCategoricalPPO requires an actor exposing "
            "`indices_to_continuous_action(obs, indices)`."
        )
        with torch.no_grad():
            cont_action = actor.indices_to_continuous_action(obs, indices)
        return cont_action

    @staticmethod
    def construct_algorithm(
        obs: TensorDict, env: VecEnv, cfg: dict, device: str
    ) -> "YahmpCategoricalPPO":
        """Mirror PPO.construct_algorithm but size storage by codebook count."""
        alg_class = resolve_callable(cfg["algorithm"].pop("class_name"))
        actor_class = resolve_callable(cfg["actor"].pop("class_name"))
        critic_class = resolve_callable(cfg["critic"].pop("class_name"))

        default_sets = ["actor", "critic"]
        if (
            "rnd_cfg" in cfg["algorithm"]
            and cfg["algorithm"]["rnd_cfg"] is not None
        ):
            default_sets.append("rnd_state")
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)

        cfg["algorithm"] = resolve_rnd_config(
            cfg["algorithm"], obs, cfg["obs_groups"], env
        )
        cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

        # Build actor/critic with the same kwargs as base PPO.
        actor: MLPModel = actor_class(
            obs, cfg["obs_groups"], "actor", env.num_actions, **cfg["actor"]
        ).to(device)
        print(f"Actor Model: {actor}")
        if cfg["algorithm"].pop("share_cnn_encoders", None):
            cfg["critic"]["cnns"] = actor.cnns  # type: ignore[attr-defined]
        critic: MLPModel = critic_class(
            obs, cfg["obs_groups"], "critic", 1, **cfg["critic"]
        ).to(device)
        print(f"Critic Model: {critic}")

        # Storage holds per-step sampled RVQ indices (one int per codebook layer),
        # NOT the continuous joint action. Size = num_active_codebooks.
        if not hasattr(actor, "num_active_codebooks"):
            raise TypeError(
                "YahmpCategoricalPPO requires an actor exposing "
                "`num_active_codebooks`."
            )
        actions_shape = [int(actor.num_active_codebooks)]
        storage = RolloutStorage(
            "rl",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            actions_shape,
            device,
        )

        alg = alg_class(
            actor,
            critic,
            storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        return alg
