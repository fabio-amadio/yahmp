"""Evaluate a YAHMP policy with GPU-parallel mjlab environments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mjlab
import numpy as np
import onnx
import onnxruntime as ort
import torch
import tyro

from yahmp.config.g1.env_cfgs import G1_EVALUATION_KEY_BODY_NAMES
from yahmp.scripts.evaluation.eval_utils import (
  GpuTrackingEvalConfig,
  build_gpu_eval_env,
  run_gpu_tracking_evaluation,
)

REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_ONNX_PATH = REPO_ROOT / "assets/models/g1_yahmp.onnx"


@dataclass(frozen=True)
class EvaluateYahmpGpuParallelConfig:
  task_id: str = "Mjlab-YAHMP-Unitree-G1"
  onnx_path: str = str(DEFAULT_ONNX_PATH)
  output_dir: str = str(REPO_ROOT / "assets/logs/yahmp_eval")
  device: str = "cuda:0"
  num_envs: int = 256


def _onnx_with_dynamic_batch(onnx_path: Path) -> bytes:
  """Return an ONNX graph whose policy batch dimension is dynamic."""
  model = onnx.load(str(onnx_path))
  if len(model.graph.input) != 1 or len(model.graph.output) != 1:
    raise ValueError("Expected a YAHMP ONNX model with one input and one output.")

  reshape_shape_inputs = {
    node.input[1]
    for node in model.graph.node
    if node.op_type == "Reshape" and len(node.input) > 1
  }
  for node in model.graph.node:
    if node.op_type != "Constant" or not set(node.output) & reshape_shape_inputs:
      continue
    value_attr = next((attr for attr in node.attribute if attr.name == "value"), None)
    if value_attr is None:
      continue
    shape = onnx.numpy_helper.to_array(value_attr.t)
    if shape.ndim == 1 and shape.size > 0 and int(shape[0]) == 1:
      dynamic_shape = np.asarray(shape, dtype=np.int64).copy()
      dynamic_shape[0] = 0
      value_attr.t.CopyFrom(onnx.numpy_helper.from_array(dynamic_shape))

  model.graph.input[0].type.tensor_type.shape.dim[0].dim_param = "batch_size"
  model.graph.output[0].type.tensor_type.shape.dim[0].dim_param = "batch_size"
  onnx.checker.check_model(model)
  return model.SerializeToString()


class YahmpOnnxPolicy:
  def __init__(self, onnx_path: Path, *, device: str, num_envs: int) -> None:
    if not device.startswith("cuda"):
      raise ValueError("ONNX GPU evaluation requires a CUDA device.")
    available = ort.get_available_providers()
    if "CUDAExecutionProvider" not in available:
      raise RuntimeError(
        "ONNX Runtime CUDAExecutionProvider is unavailable. "
        f"Available providers: {available}"
      )

    self.session = ort.InferenceSession(
      _onnx_with_dynamic_batch(onnx_path),
      providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
    )
    self.input = self.session.get_inputs()[0]
    self.output = self.session.get_outputs()[0]
    self.device_id = int(device.split(":", maxsplit=1)[1]) if ":" in device else 0
    self.num_envs = num_envs
    self.observation_dim = int(self.input.shape[1])
    self.action_dim = int(self.output.shape[1])
    self.output_actions = torch.empty(
      (num_envs, self.action_dim), device=device, dtype=torch.float32
    )

  @torch.inference_mode()
  def act(self, observations: dict[str, torch.Tensor]) -> torch.Tensor:
    obs = observations["actor"].contiguous()
    if obs.shape != (self.num_envs, self.observation_dim):
      raise ValueError(
        f"Expected actor observations {(self.num_envs, self.observation_dim)}, "
        f"got {tuple(obs.shape)}. Check that `--task-id` matches the ONNX policy."
      )

    binding = self.session.io_binding()
    binding.bind_input(
      self.input.name,
      "cuda",
      self.device_id,
      np.float32,
      tuple(obs.shape),
      obs.data_ptr(),
    )
    binding.bind_output(
      self.output.name,
      "cuda",
      self.device_id,
      np.float32,
      tuple(self.output_actions.shape),
      self.output_actions.data_ptr(),
    )
    self.session.run_with_iobinding(binding)
    return self.output_actions


def run(cfg: EvaluateYahmpGpuParallelConfig) -> dict:
  import mjlab.tasks  # noqa: F401

  import yahmp.config.g1  # noqa: F401

  onnx_path = Path(cfg.onnx_path).expanduser().resolve()
  if not onnx_path.is_file():
    raise FileNotFoundError(f"ONNX policy not found: {onnx_path}")

  eval_cfg = GpuTrackingEvalConfig(
    task_id=cfg.task_id,
    output_dir=cfg.output_dir,
    device=cfg.device,
    num_envs=cfg.num_envs,
    key_body_names=G1_EVALUATION_KEY_BODY_NAMES,
  )
  env = build_gpu_eval_env(eval_cfg)
  try:
    policy = YahmpOnnxPolicy(
      onnx_path,
      device=cfg.device,
      num_envs=env.num_envs,
    )
    return run_gpu_tracking_evaluation(
      eval_cfg,
      env,
      policy.act,
      policy_metadata={
        "task_id": cfg.task_id,
        "onnx_path": str(onnx_path),
      },
    )
  finally:
    env.close()


def main() -> None:
  cfg = tyro.cli(EvaluateYahmpGpuParallelConfig, config=mjlab.TYRO_FLAGS)
  result = run(cfg)
  print(
    json.dumps(
      {key: value for key, value in result.items() if key != "summary"}, indent=2
    )
  )


if __name__ == "__main__":
  main()
