from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.utils import resolve_callable

from yahmp.mdp.motion.base import MotionCommand


class YahmpImitationRunner:
    """Phase 1 runner: train `YahmpImitationModel` by distilling a frozen expert.

    Each iteration:
      1. Roll out the frozen encoder-decoder expert in the env for
         ``num_steps_per_env`` steps, recording (obs, expert_action) at each step.
         The expert's deterministic action is used to step the env, so the
         collected state distribution is the expert's on-policy distribution.
      2. Update the `YahmpImitationModel` on the collected (obs, expert_action)
         batch using `ImitationTrainer`. No PPO, no value function — pure
         supervised imitation with the RVQ commitment loss.

    The runner expects the following keys in ``train_cfg``:
      - ``expert``: dict with ``class_name`` plus model kwargs for the expert.
      - ``student``: dict with ``class_name`` plus model kwargs for the student.
      - ``expert_checkpoint``: path to the trained encoder-decoder checkpoint.
      - ``loss_weights``: kwargs for `ImitationLossWeights`.
      - ``trainer``: kwargs for `ImitationTrainerCfg`.
      - ``num_steps_per_env``: rollout length per iteration.
      - ``save_interval``: iterations between checkpoint saves.
      - ``obs_groups``: dict matching the on-policy runner format.
    """

    env: RslRlVecEnvWrapper

    def __init__(
        self,
        env: Any,
        train_cfg: dict,
        log_dir: str | None = None,
        device: str = "cuda",
        **kwargs: Any,
    ) -> None:
        del kwargs  # absorb mjlab.scripts.train extras like `registry_name`
        if not isinstance(env, RslRlVecEnvWrapper):
            raise TypeError(
                "YahmpImitationRunner requires an RslRlVecEnvWrapper-wrapped env."
            )
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.log_dir = log_dir

        self._configure_model_cfg(env, train_cfg)

        obs = env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]

        obs_groups = train_cfg.get("obs_groups", {"actor": ("actor",)})
        action_dim = int(env.num_actions)

        expert_cfg = dict(train_cfg["expert"])
        expert_cls = resolve_callable(expert_cfg.pop("class_name"))
        expert_cfg.pop(
            "cnn_cfg", None
        )  # expert doesn't use the RSL private CNN, so ignore any cnn_cfg
        self.expert = expert_cls(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="actor",
            output_dim=action_dim,
            **expert_cfg,
        ).to(device)

        expert_ckpt = train_cfg.get("expert_checkpoint")
        if expert_ckpt is not None:
            self._load_expert(expert_ckpt)
        self.expert.eval()
        for p in self.expert.parameters():
            p.requires_grad = False

        student_cfg = dict(train_cfg["student"])
        student_cls = resolve_callable(student_cfg.pop("class_name"))
        student_cfg.pop(
            "cnn_cfg", None
        )  # student doesn't use the RSL private CNN, so ignore any cnn_cfg
        self.student = student_cls(
            obs=obs,
            obs_groups=obs_groups,
            obs_set="actor",
            output_dim=action_dim,
            **student_cfg,
        ).to(device)

        # Build the trainer (imported here to avoid hard cycles at module import).
        from yahmp.rl.imitation_trainer import (
            ImitationLossWeights,
            ImitationTrainer,
            ImitationTrainerCfg,
        )

        loss_weights = ImitationLossWeights(**train_cfg.get("loss_weights", {}))
        trainer_cfg = ImitationTrainerCfg(**train_cfg.get("trainer", {}))
        self.trainer = ImitationTrainer(
            model=self.student,
            loss_weights=loss_weights,
            cfg=trainer_cfg,
            device=device,
        )

        self.num_steps_per_env = int(train_cfg.get("num_steps_per_env", 24))
        self.save_interval = int(train_cfg.get("save_interval", 500))
        self.current_iteration = 0

        self._wandb_initialized = False
        self._use_wandb = (
            train_cfg.get("logger", "wandb") == "wandb" and log_dir is not None
        )

    def _load_expert(self, ckpt_path: str | os.PathLike) -> None:
        print(f"[YahmpImitationRunner] Loading expert checkpoint: {ckpt_path}")
        ckpt = torch.load(str(ckpt_path), map_location=self.device, weights_only=False)
        sd = ckpt.get("actor_state_dict", ckpt.get("model_state_dict", ckpt))
        if "std" in sd:
            sd["distribution.std_param"] = sd.pop("std")
        if "log_std" in sd:
            sd["distribution.log_std_param"] = sd.pop("log_std")
        missing, unexpected = self.expert.load_state_dict(sd, strict=False)
        print(
            f"[YahmpImitationRunner] Expert loaded — missing={len(missing)} "
            f"unexpected={len(unexpected)}"
        )

    @torch.no_grad()
    def _expert_action(self, obs) -> torch.Tensor:
        """Deterministic mean action from the expert."""
        return self.expert(obs)

    def add_git_repo_to_log(self, *args, **kwargs) -> None:
        pass

    def _maybe_init_wandb(self) -> None:
        if not self._use_wandb or self._wandb_initialized:
            return
        try:
            import wandb
        except ImportError:
            print("[YahmpImitationRunner] wandb not installed — skipping logging.")
            self._use_wandb = False
            return
        entity = os.environ.get("WANDB_USERNAME") or os.environ.get("WANDB_ENTITY")
        wandb.init(
            project=self.cfg.get("wandb_project", "yahmp"),
            entity=entity,
            name=Path(self.log_dir).name if self.log_dir else None,
            tags=list(self.cfg.get("wandb_tags", ())),
            config=self.cfg,
        )
        self._wandb_initialized = True

    def learn(self, num_learning_iterations: int, **kwargs: Any) -> None:
        del kwargs  # absorb `init_at_random_ep_len` etc.
        self._maybe_init_wandb()
        obs = self.env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = obs.to(self.device)

        num_envs = int(self.env.num_envs)
        cur_ep_len = torch.zeros(num_envs, device=self.device, dtype=torch.long)

        target_iteration = self.current_iteration + int(num_learning_iterations)
        while self.current_iteration < target_iteration:
            rollout_obs: list = []
            rollout_actions: list[torch.Tensor] = []
            rollout_dones: list[torch.Tensor] = []
            ended_lengths: list[torch.Tensor] = []

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    action = self._expert_action(obs)
                    rollout_obs.append(obs.clone())
                    rollout_actions.append(action.clone())
                    obs, _, dones, _ = self.env.step(action.to(self.env.device))
                    dones = dones.to(self.device)
                    rollout_dones.append(dones.clone())
                    cur_ep_len += 1
                    done_mask = dones.bool()
                    if done_mask.any():
                        ended_lengths.append(cur_ep_len[done_mask].clone())
                        cur_ep_len[done_mask] = 0
                    obs = obs.to(self.device)

            stats: dict[str, float] = {}
            for i, (step_obs, step_action) in enumerate(
                zip(rollout_obs, rollout_actions, strict=True)
            ):
                step_obs = step_obs.clone()
                step_action = step_action.clone()
                batch: dict = {"obs": step_obs, "a_expert": step_action}
                if i > 0:
                    batch["valid_prev"] = rollout_dones[i - 1]
                stats = self.trainer.train_step(batch)

            if ended_lengths:
                lens = torch.cat(ended_lengths).float()
                stats["expert/episode_length_mean"] = float(lens.mean().item())
                stats["expert/episode_length_max"] = float(lens.max().item())
                stats["expert/num_episodes"] = float(lens.numel())
            else:
                stats["expert/episode_length_mean"] = float(cur_ep_len.float().mean().item())
                stats["expert/episode_length_max"] = float(cur_ep_len.max().item())
                stats["expert/num_episodes"] = 0.0

            self.current_iteration += 1
            it = self.current_iteration
            print(
                f"[iter {it}] "
                f"loss_total={stats.get('loss_total', float('nan')):.4f} "
                f"loss_action={stats.get('loss_action', float('nan')):.4f} "
                f"loss_mm={stats.get('loss_mm', float('nan')):.4f} "
                f"loss_commit={stats.get('loss_commit', float('nan')):.4f}"
            )

            if self._use_wandb and self._wandb_initialized:
                import wandb

                wandb.log(stats, step=it)

            if self.log_dir and it % self.save_interval == 0:
                self.save(os.path.join(self.log_dir, f"model_{it}.pt"))

        if self.log_dir:
            self.save(
                os.path.join(self.log_dir, f"model_{self.current_iteration}.pt")
            )

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trainer": self.trainer.state_dict(),
                "iteration": self.current_iteration,
            },
            path,
        )
        if self._use_wandb and self._wandb_initialized:
            import wandb
            wandb.save(path, base_path=str(Path(path).parent), policy="now")

    def load(self, path: str, strict: bool = True, **kwargs: Any) -> None:
        del kwargs  # absorb mjlab.scripts.play extras like `load_cfg`, `map_location`
        sd = torch.load(path, map_location=self.device, weights_only=False)
        self.trainer.load_state_dict(sd["trainer"], strict=strict)
        self.current_iteration = int(sd.get("iteration", 0))

    def get_inference_policy(self, device: str | torch.device | None = None):
        """Return a callable mapping obs -> action for use in `play`."""
        if device is not None:
            self.student.to(device)
        self.student.eval()

        @torch.inference_mode()
        def policy(obs):
            out = self.student(obs)
            return out["a_hat"] if isinstance(out, dict) else out

        return policy

    @staticmethod
    def _configure_model_cfg(env: Any, train_cfg: dict) -> None:
        """Inject env-derived dims into expert/student configs.

        Mirrors the dimension logic of `YahmpOnPolicyRunner._configure_model_cfg`
        for the encoder-decoder variant, applied to both ``expert`` and
        ``student`` blocks.
        """
        if not isinstance(env, RslRlVecEnvWrapper):
            return
        env_unwrapped = env.unwrapped
        motion_term = env_unwrapped.command_manager.get_term("motion")
        if not isinstance(motion_term, MotionCommand):
            return

        observation_manager = env_unwrapped.observation_manager
        obs_dims = observation_manager.group_obs_dim
        actor_group = "actor"
        actor_terms = observation_manager.active_terms[actor_group]
        actor_term_dims = observation_manager._group_obs_term_dim[actor_group]
        flat_term_dims = {
            name: int(math.prod(dims))
            for name, dims in zip(actor_terms, actor_term_dims, strict=False)
        }
        command_dim = flat_term_dims["command"]
        history_dim = flat_term_dims["history"]
        current_dim = int(obs_dims[actor_group][0]) - command_dim - history_dim
        history_step_dim = command_dim + current_dim
        if history_dim % history_step_dim != 0:
            raise ValueError(
                "YahmpImitationRunner history dimension mismatch: "
                f"history_dim={history_dim}, history_step_dim={history_step_dim}."
            )
        history_steps = max(history_dim // history_step_dim, 1)

        for key in ("expert", "student"):
            if key not in train_cfg:
                continue
            cfg = train_cfg[key]
            cfg.setdefault("current_motion_obs_dim", command_dim)
            cfg.setdefault("proprio_obs_dim", current_dim)
            cfg.setdefault("history_steps", history_steps)
            cfg.setdefault("history_latent_dim", 128)
            cfg.setdefault("history_conv_channels", (64, 32))
            cfg.setdefault("history_conv_kernel_sizes", (4, 2))
            cfg.setdefault("history_conv_strides", (2, 1))
            cfg.setdefault("layer_norm", True)
            class_name = str(cfg.get("class_name", ""))
            if "EncoderDecoder" in class_name:
                cfg.setdefault("latent_dim", 128)
                cfg.setdefault("encoder_hidden_dims", (512, 512, 256, 128))
            if "Imitation" in class_name:
                cfg.setdefault("latent_dim", 128)
                cfg.setdefault("prior_hidden_dims", (512, 512, 256, 128))
                cfg.setdefault("posterior_hidden_dims", (512, 512, 256, 128))
                cfg.setdefault("hidden_dims", (512, 512, 256, 128))
                cfg.setdefault("activation", "elu")
                cfg.setdefault("obs_normalization", True)
