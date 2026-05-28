from __future__ import annotations

import math
import os
from pathlib import Path
from typing import Any

import torch
from mjlab.rl import RslRlVecEnvWrapper
from rsl_rl.utils import resolve_callable

from yahmp.mdp.actions import ResidualJointPositionAction
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
      - ``action_target_mode``: ``expert_residual`` for legacy residual targets,
        or ``default_offset`` to train the student in locomotion action space.
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

        self.action_target_mode = str(
            train_cfg.get("action_target_mode", "default_offset")
        )
        self._motion_command: MotionCommand | None = None
        self._joint_action_term: Any | None = None
        self._actor_action_slice: slice | None = None
        self._actor_history_slice: slice | None = None
        self._history_action_slice: slice | None = None
        self._history_steps: int = 0
        self._history_step_dim: int = 0
        self._configure_action_target_conversion()

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

    def _configure_action_target_conversion(self) -> None:
        valid_modes = {"expert_residual", "default_offset"}
        if self.action_target_mode not in valid_modes:
            raise ValueError(
                f"Unsupported action_target_mode={self.action_target_mode!r}. "
                f"Expected one of {sorted(valid_modes)}."
            )
        if self.action_target_mode == "expert_residual":
            print("[YahmpImitationRunner] Using legacy expert-residual targets.")
            return

        motion_term = self.env.unwrapped.command_manager.get_term("motion")
        if not isinstance(motion_term, MotionCommand):
            raise TypeError(
                "default_offset imitation targets require a MotionCommand named "
                f"'motion', got {type(motion_term)}."
            )

        action_term = self.env.unwrapped.action_manager.get_term("joint_pos")
        for attr in ("target_ids", "scale"):
            if not hasattr(action_term, attr):
                raise TypeError(
                    "default_offset imitation targets require a joint-position "
                    f"action term exposing `{attr}`; got {type(action_term)}."
                )

        self._motion_command = motion_term
        self._joint_action_term = action_term
        self._configure_default_offset_observation_rewrite()
        print(
            "[YahmpImitationRunner] Using default-offset imitation targets "
            "compatible with locomotion actions."
        )

    def _configure_default_offset_observation_rewrite(self) -> None:
        observation_manager = self.env.unwrapped.observation_manager
        actor_group = "actor"
        actor_terms = observation_manager.active_terms[actor_group]
        actor_term_dims = observation_manager._group_obs_term_dim[actor_group]
        flat_term_dims = {
            name: int(math.prod(dims))
            for name, dims in zip(actor_terms, actor_term_dims, strict=False)
        }

        if "actions" not in flat_term_dims or "history" not in flat_term_dims:
            raise ValueError(
                "default_offset imitation targets require actor observations "
                "with `actions` and `history` terms so last_action semantics can "
                "be rewritten."
            )

        offset = 0
        term_slices: dict[str, slice] = {}
        for name in actor_terms:
            dim = flat_term_dims[name]
            term_slices[name] = slice(offset, offset + dim)
            offset += dim

        action_dim = int(self.env.num_actions)
        history_dim = flat_term_dims["history"]
        current_dim = offset - flat_term_dims["command"] - history_dim
        current_start = term_slices["command"].stop
        if history_dim % current_dim != 0:
            raise ValueError(
                "default_offset observation rewrite history dimension mismatch: "
                f"history_dim={history_dim}, current_dim={current_dim}."
            )
        action_slice = term_slices["actions"]
        if action_slice.stop - action_slice.start != action_dim:
            raise ValueError(
                "default_offset observation rewrite action dimension mismatch: "
                f"obs_actions={action_slice.stop - action_slice.start}, "
                f"env_actions={action_dim}."
            )
        if action_slice.start < current_start or action_slice.stop > term_slices["history"].start:
            raise ValueError(
                "default_offset observation rewrite expects `actions` inside the "
                "current proprio block between `command` and `history`."
            )

        self._actor_action_slice = action_slice
        self._actor_history_slice = term_slices["history"]
        self._history_steps = max(history_dim // current_dim, 1)
        self._history_step_dim = current_dim
        self._history_action_slice = slice(
            action_slice.start - current_start,
            action_slice.stop - current_start,
        )

    def _assert_learn_env_supports_expert_rollout(self) -> None:
        if self.action_target_mode != "default_offset":
            return
        if isinstance(self._joint_action_term, ResidualJointPositionAction):
            return
        raise RuntimeError(
            "default_offset imitation training must roll out the expert in a "
            "residual-action env, then convert the supervised target. Do not train "
            "the `NoRes` imitation task; use it only for play/evaluation."
        )

    @staticmethod
    def _action_affine_tensor(value: Any, like: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            return value.to(device=like.device, dtype=like.dtype)
        return torch.as_tensor(value, device=like.device, dtype=like.dtype)

    @torch.no_grad()
    def _imitation_action_target(self, expert_action: torch.Tensor) -> torch.Tensor:
        """Return the supervised student target for the current expert action."""
        if self.action_target_mode == "expert_residual":
            return expert_action

        assert self._motion_command is not None
        assert self._joint_action_term is not None

        target_ids = self._joint_action_term.target_ids
        q_ref = self._motion_command.joint_pos[:, target_ids].to(
            device=expert_action.device, dtype=expert_action.dtype
        )
        q_home = self._joint_action_term._entity.data.default_joint_pos[
            :, target_ids
        ].to(device=expert_action.device, dtype=expert_action.dtype)

        scale = self._action_affine_tensor(self._joint_action_term.scale, expert_action)
        offset = self._action_affine_tensor(
            getattr(self._joint_action_term, "offset", 0.0), expert_action
        )
        eps = torch.finfo(expert_action.dtype).eps
        if torch.any(torch.abs(scale) <= eps):
            raise ValueError("Cannot convert imitation targets with zero action scale.")

        expert_delta_q = expert_action * scale + offset
        expert_target_q = q_ref + expert_delta_q
        return (expert_target_q - q_home) / scale

    def _student_observation(
        self,
        obs,
        last_default_action: torch.Tensor,
        default_action_history: torch.Tensor,
    ):
        """Return obs with last_action slots rewritten to student action semantics."""
        if self.action_target_mode == "expert_residual":
            return obs

        assert self._actor_action_slice is not None
        assert self._actor_history_slice is not None
        assert self._history_action_slice is not None

        student_obs = obs.clone()
        actor_obs = student_obs["actor"].clone()
        last_default_action = last_default_action.to(
            device=actor_obs.device, dtype=actor_obs.dtype
        )
        default_action_history = default_action_history.to(
            device=actor_obs.device, dtype=actor_obs.dtype
        )

        actor_obs[:, self._actor_action_slice] = last_default_action
        history = actor_obs[:, self._actor_history_slice].reshape(
            actor_obs.shape[0], self._history_steps, self._history_step_dim
        )
        history[:, :, self._history_action_slice] = default_action_history
        actor_obs[:, self._actor_history_slice] = history.reshape(
            actor_obs.shape[0], -1
        )
        student_obs["actor"] = actor_obs
        return student_obs

    def _update_default_action_history(
        self,
        default_action_history: torch.Tensor,
        last_default_action: torch.Tensor,
    ) -> None:
        if self.action_target_mode == "expert_residual":
            return
        default_action_history[:, :-1] = default_action_history[:, 1:].clone()
        default_action_history[:, -1] = last_default_action

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
        self._assert_learn_env_supports_expert_rollout()
        self._maybe_init_wandb()
        obs = self.env.get_observations()
        if isinstance(obs, tuple):
            obs = obs[0]
        obs = obs.to(self.device)

        num_envs = int(self.env.num_envs)
        cur_ep_len = torch.zeros(num_envs, device=self.device, dtype=torch.long)
        last_default_action = torch.zeros(
            num_envs, int(self.env.num_actions), device=self.device
        )
        default_action_history = torch.zeros(
            num_envs,
            max(self._history_steps, 1),
            int(self.env.num_actions),
            device=self.device,
        )

        target_iteration = self.current_iteration + int(num_learning_iterations)
        while self.current_iteration < target_iteration:
            rollout_obs: list = []
            rollout_actions: list[torch.Tensor] = []
            rollout_dones: list[torch.Tensor] = []
            ended_lengths: list[torch.Tensor] = []

            with torch.inference_mode():
                for _ in range(self.num_steps_per_env):
                    student_obs = self._student_observation(
                        obs, last_default_action, default_action_history
                    )
                    self._update_default_action_history(
                        default_action_history, last_default_action
                    )
                    action = self._expert_action(obs)
                    action_target = self._imitation_action_target(action)
                    rollout_obs.append(student_obs.clone())
                    rollout_actions.append(action_target.clone())
                    obs, _, dones, _ = self.env.step(action.to(self.env.device))
                    dones = dones.to(self.device)
                    rollout_dones.append(dones.clone())
                    cur_ep_len += 1
                    last_default_action = action_target.detach().clone()
                    done_mask = dones.bool()
                    if done_mask.any():
                        ended_lengths.append(cur_ep_len[done_mask].clone())
                        cur_ep_len[done_mask] = 0
                        last_default_action[done_mask] = 0.0
                        default_action_history[done_mask] = 0.0
                    obs = obs.to(self.device)

            stats: dict[str, float] = {}
            for i, (step_obs, step_action_target) in enumerate(
                zip(rollout_obs, rollout_actions, strict=True)
            ):
                step_obs = step_obs.clone()
                step_action_target = step_action_target.clone()
                batch: dict = {"obs": step_obs, "a_expert": step_action_target}
                if i > 0:
                    batch["valid_prev"] = rollout_dones[i - 1]
                stats = self.trainer.train_step(batch)

            action_targets = torch.stack(rollout_actions)
            stats["target/raw_mean"] = float(action_targets.mean().item())
            stats["target/raw_std"] = float(action_targets.std().item())
            stats["target/raw_abs_max"] = float(action_targets.abs().max().item())

            if ended_lengths:
                lens = torch.cat(ended_lengths).float()
                stats["expert/episode_length_mean"] = float(lens.mean().item())
                stats["expert/episode_length_max"] = float(lens.max().item())
                stats["expert/num_episodes"] = float(lens.numel())
            else:
                stats["expert/episode_length_mean"] = float(
                    cur_ep_len.float().mean().item()
                )
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
            self.save(os.path.join(self.log_dir, f"model_{self.current_iteration}.pt"))

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "trainer": self.trainer.state_dict(),
                "iteration": self.current_iteration,
                "action_target_mode": self.action_target_mode,
            },
            path,
        )
        if self._use_wandb and self._wandb_initialized:
            import wandb

            wandb.save(path, base_path=str(Path(path).parent), policy="now")

    def load(self, path: str, strict: bool = True, **kwargs: Any) -> None:
        del kwargs  # absorb mjlab.scripts.play extras like `load_cfg`, `map_location`
        sd = torch.load(path, map_location=self.device, weights_only=False)
        checkpoint_mode = sd.get("action_target_mode")
        if checkpoint_mode is None and self.action_target_mode == "default_offset":
            raise ValueError(
                "This runner expects a default-offset imitation checkpoint, but "
                "the checkpoint has no `action_target_mode` metadata. Retrain the "
                "imitation with the current runner before using the NoRes play task "
                "or locomotion."
            )
        if checkpoint_mode is not None and str(checkpoint_mode) != self.action_target_mode:
            raise ValueError(
                "Imitation checkpoint action semantics mismatch: "
                f"checkpoint={checkpoint_mode!r}, runner={self.action_target_mode!r}."
            )
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
        history_step_dim = current_dim
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
