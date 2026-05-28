from dataclasses import dataclass, field
from typing import Literal

from mjlab.rl import (
  RslRlBaseRunnerCfg,
  RslRlModelCfg,
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

from yahmp.rl.imitation_trainer import ImitationLossWeights, ImitationTrainerCfg

UploadModelMode = Literal["all", "rolling_latest"]
ImitationActionTargetMode = Literal["expert_residual", "default_offset"]


@dataclass
class YahmpDistillationAlgorithmCfg:
  class_name: str = "rsl_rl.algorithms:Distillation"
  num_learning_epochs: int = 1
  gradient_length: int = 15
  learning_rate: float = 1e-3
  max_grad_norm: float | None = 1.0
  loss_type: Literal["mse", "huber"] = "mse"
  optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
  rnd_cfg: dict | None = None
  symmetry_cfg: dict | None = None


@dataclass
class YahmpOnPolicyRunnerCfg(RslRlOnPolicyRunnerCfg):
  upload_model_mode: UploadModelMode = "rolling_latest"


@dataclass
class YahmpStudentOnPolicyRunnerCfg(YahmpOnPolicyRunnerCfg):
  teacher_wandb_run_path: str | None = None
  teacher_wandb_checkpoint_name: str | None = None
  teacher_checkpoint_file: str | None = None
  teacher_strict_load: bool = True


@dataclass
class YahmpLocomotionOnPolicyRunnerCfg(YahmpOnPolicyRunnerCfg):
  """PPO runner config for the YAHMP locomotion task.

  Loads a trained ``YahmpImitationModel`` checkpoint and freezes its
  prior + Residual VQ + action_decoder inside the locomotion actor.
  """

  imitation_checkpoint_file: str | None = None
  imitation_wandb_run_path: str | None = None
  imitation_wandb_checkpoint_name: str | None = None
  imitation_strict_load: bool = True
  imitation_copy_normalizer_proprio_history: bool = True


@dataclass
class YahmpActionMatchingPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "yahmp.rl.action_matching_ppo_algorithm:YahmpActionMatchingPPO"
  bc_coef_start: float = 1.0
  bc_coef_end: float = 0.0
  bc_anneal_iters: int = 10_000
  bc_loss_type: Literal["mse", "huber"] = "mse"


@dataclass
class YahmpKlMatchingPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "yahmp.rl.kl_matching_ppo_algorithm:YahmpKlMatchingPPO"
  kl_coef: float = 0.1
  kl_coef_min: float = 0.0
  kl_coef_anneal_iters: int = 10_000


@dataclass
class YahmpImitationRunnerCfg(RslRlBaseRunnerCfg):
  """Config for the Phase 1 imitation/VQ runner.

  Inherits the standard mjlab base runner fields (seed, max_iterations,
  experiment_name, clip_actions, …) so it plugs into ``mjlab.scripts.train``
  the same way the PPO runner does. The runner-specific fields below describe
  the frozen expert, the student model, and the supervised loss / optimizer.

  ``expert_checkpoint`` must be provided (CLI: ``--agent.expert-checkpoint``);
  it points to the trained encoder-decoder policy that supplies target
  actions during rollouts.
  """

  class_name: str = "yahmp.rl.imitation_runner:YahmpImitationRunner"
  expert: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="yahmp.rl.policy:YahmpEncoderDecoderActorModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
      distribution_cfg={
        "class_name": "GaussianDistribution",
        "init_std": 1.0,
        "std_type": "log",
      },
    )
  )
  student: RslRlModelCfg = field(
    default_factory=lambda: RslRlModelCfg(
      class_name="yahmp.rl.imitation_RVQ_policy:YahmpImitationModel",
      hidden_dims=(512, 512, 256, 128),
      activation="elu",
      obs_normalization=True,
    )
  )
  expert_checkpoint: str | None = None
  action_target_mode: ImitationActionTargetMode = "default_offset"
  loss_weights: ImitationLossWeights = field(default_factory=ImitationLossWeights)
  trainer: ImitationTrainerCfg = field(default_factory=ImitationTrainerCfg)
  upload_model_mode: UploadModelMode = "rolling_latest"
