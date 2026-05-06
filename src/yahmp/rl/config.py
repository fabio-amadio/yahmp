from dataclasses import dataclass
from typing import Literal

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

UploadModelMode = Literal["all", "rolling_latest"]


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
