from dataclasses import dataclass
from typing import Literal

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoAlgorithmCfg,
)

UploadModelMode = Literal["all", "rolling_latest"]


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
class YahmpKlMatchingPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
  class_name: str = "yahmp.rl.kl_matching_ppo_algorithm:YahmpKlMatchingPPO"
  kl_coef: float = 0.1
  kl_coef_min: float = 0.0
  kl_coef_anneal_iters: int = 10_000
