"""Core YAHMP motion-command state, reference queries, and shared utilities."""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch
from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  quat_error_magnitude,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

from .debug_visualizer import debug_visualize_motion_command
from .indexing import build_name_to_index, resolve_motion_body_names
from .library import MotionFrameBatch, MotionLibrary, MotionLoader
from .sampling import (
  adaptive_sampling,
  cap_sampling_probabilities,
  resample_command,
  reset_robot_to_reference,
  uniform_sampling,
  update_command,
)

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv


class MotionCommand(CommandTerm):
  """Base YAHMP motion command term."""

  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.entity_name]
    self._uses_motion_library = False
    self.root_body_name = self._resolve_root_body_name()

    self.motion: MotionLoader | None = None
    self.motion_lib: MotionLibrary | None = None
    self._current_motion_frame: MotionFrameBatch | None = None
    self._load_motion_source(self._required_motion_body_names())
    self._initialize_body_mappings()
    self._initialize_tracking_buffers()
    self._initialize_metrics()

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  def _resolve_root_body_name(self) -> str:
    """Resolve and validate the body used as the articulated root reference."""
    if self.cfg.anchor_body_name == "":
      raise ValueError("`anchor_body_name` must be set for MotionCommand.")
    if len(self.cfg.body_names) == 0:
      raise ValueError("`body_names` must be non-empty for MotionCommand.")
    root_body_name = self.cfg.root_body_name or self.cfg.body_names[0]
    if root_body_name == "":
      raise ValueError("`root_body_name` cannot be empty when provided.")
    return root_body_name

  def _required_motion_body_names(self) -> tuple[str, ...]:
    """Return the unique set of bodies that must be present in the motion data."""
    return tuple(
      dict.fromkeys(
        (self.cfg.anchor_body_name, *self.cfg.body_names, self.root_body_name)
      )
    )

  def _load_motion_source(self, required_motion_body_names: tuple[str, ...]) -> None:
    """Load either a single motion file or a multi-clip motion library."""
    source = Path(self.cfg.motion_file)
    if source.suffix == ".npz":
      if not source.is_file():
        raise ValueError(f"Motion file does not exist: {self.cfg.motion_file}")
      self.motion = MotionLoader(
        self.cfg.motion_file,
        device=self.device,
        required_body_names=required_motion_body_names,
      )
      return

    self.motion_lib = MotionLibrary(
      self.cfg.motion_file,
      device=self.device,
      show_progress=self.cfg.show_motion_load_progress,
      required_body_names=required_motion_body_names,
    )
    self._uses_motion_library = True

  def _initialize_body_mappings(self) -> None:
    """Build robot/motion body index mappings used by all command queries."""
    motion_body_names = resolve_motion_body_names(self)
    motion_name_to_index = build_name_to_index(motion_body_names, source="motion")
    robot_body_names = tuple(self.robot.body_names)
    robot_name_to_index = build_name_to_index(robot_body_names, source="robot")

    required_body_names = list(
      dict.fromkeys((self.cfg.anchor_body_name, *self.cfg.body_names))
    )
    missing_motion = [n for n in required_body_names if n not in motion_name_to_index]
    missing_robot = [n for n in required_body_names if n not in robot_name_to_index]
    if missing_motion or missing_robot:
      error_lines = ["Body name mapping failed while initializing YAHMP MotionCommand."]
      if missing_motion:
        error_lines.append(f"  Missing in motion reference: {missing_motion}")
      if missing_robot:
        error_lines.append(f"  Missing in robot model: {missing_robot}")
      raise ValueError("\n".join(error_lines))

    self.robot_anchor_body_index = robot_name_to_index[self.cfg.anchor_body_name]
    self.motion_anchor_body_index = motion_name_to_index[self.cfg.anchor_body_name]
    self.motion_root_body_index = motion_name_to_index[self.root_body_name]
    self.robot_body_indexes = torch.tensor(
      [robot_name_to_index[name] for name in self.cfg.body_names],
      dtype=torch.long,
      device=self.device,
    )
    self.motion_body_indexes = torch.tensor(
      [motion_name_to_index[name] for name in self.cfg.body_names],
      dtype=torch.long,
      device=self.device,
    )

  def _initialize_tracking_buffers(self) -> None:
    """Allocate tensors that track current reference state and sampling stats."""
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.motion_time_offsets = torch.zeros(
      self.num_envs, dtype=torch.float32, device=self.device
    )

    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(self.cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(self.cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    if self._uses_motion_library:
      assert self.motion_lib is not None
      max_motion_len_s = float(torch.max(self.motion_lib.motion_lengths_s).item())
      self.bin_count = max(int(max_motion_len_s / max(self._env.step_dt, 1e-6)) + 1, 1)
      self._refresh_motion_frame()
    else:
      assert self.motion is not None
      self.bin_count = int(self.motion.time_step_total // (1 / self._env.step_dt)) + 1

    self.bin_failed_count: torch.Tensor | None = None
    self._current_bin_failed: torch.Tensor | None = None
    self.motion_failed_count: torch.Tensor | None = None
    self._current_motion_failed: torch.Tensor | None = None
    self.phase_failed_count: torch.Tensor | None = None
    self._current_phase_failed: torch.Tensor | None = None

    if self._uses_motion_library:
      assert self.motion_lib is not None
      num_motions = self.motion_lib.num_motions()
      self.motion_failed_count = torch.zeros(
        num_motions, dtype=torch.float, device=self.device
      )
      self._current_motion_failed = torch.zeros(
        num_motions, dtype=torch.float, device=self.device
      )
      self.phase_failed_count = torch.zeros(
        (num_motions, self.bin_count), dtype=torch.float, device=self.device
      )
      self._current_phase_failed = torch.zeros(
        (num_motions, self.bin_count), dtype=torch.float, device=self.device
      )
      self.motion_completion_ema = torch.zeros(
        num_motions, dtype=torch.float, device=self.device
      )
      self.motion_attempt_count = torch.zeros(
        num_motions, dtype=torch.long, device=self.device
      )
    else:
      self.bin_failed_count = torch.zeros(
        self.bin_count, dtype=torch.float, device=self.device
      )
      self._current_bin_failed = torch.zeros(
        self.bin_count, dtype=torch.float, device=self.device
      )

    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

  def _initialize_metrics(self) -> None:
    """Create all command metrics eagerly so logging keys stay stable."""
    metric_names = (
      "error_anchor_pos",
      "error_anchor_rot",
      "error_anchor_lin_vel",
      "error_anchor_ang_vel",
      "error_body_pos",
      "error_body_rot",
      "error_body_lin_vel",
      "error_body_ang_vel",
      "error_joint_pos",
      "error_joint_vel",
      "sampling_motion_entropy",
      "sampling_motion_top1_prob",
      "sampling_motion_top1_idx",
      "sampling_phase_entropy",
      "sampling_phase_top1_prob",
      "sampling_phase_top1_bin",
    )
    self._metric_episode_sums: dict[str, torch.Tensor] = {}
    self._metric_step_counts = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )
    for metric_name in metric_names:
      self.metrics[metric_name] = torch.zeros(self.num_envs, device=self.device)
      self._metric_episode_sums[metric_name] = torch.zeros(
        self.num_envs, device=self.device
      )

  def reset(self, env_ids: torch.Tensor | slice | None) -> dict[str, float]:
    """Reset command state and log per-episode averages for motion metrics."""
    assert isinstance(env_ids, torch.Tensor)
    self._update_motion_completion_stats(env_ids)
    extras: dict[str, float] = {}
    counts = self._metric_step_counts[env_ids].to(torch.float32)
    safe_counts = torch.clamp(counts, min=1.0)

    for metric_name, metric_sum in self._metric_episode_sums.items():
      episode_avg = torch.mean(metric_sum[env_ids] / safe_counts)
      extras[metric_name] = episode_avg.item()
      metric_sum[env_ids] = 0.0
      self.metrics[metric_name][env_ids] = 0.0

    self._metric_step_counts[env_ids] = 0
    self.command_counter[env_ids] = 0
    self._resample(env_ids)
    reset_robot_to_reference(self, env_ids)
    return extras

  def _update_motion_completion_stats(self, env_ids: torch.Tensor) -> None:
    """Track a per-motion EMA of the reached reference phase during training."""
    if not self._uses_motion_library:
      return
    assert self.motion_lib is not None
    if not hasattr(self, "motion_completion_ema"):
      return

    episode_lengths = self._env.episode_length_buf[env_ids]
    valid_mask = episode_lengths > 0
    if not torch.any(valid_mask):
      return

    valid_env_ids = env_ids[valid_mask]
    motion_ids = self.motion_ids[valid_env_ids]
    motion_lengths = self.motion_lib.get_motion_length(motion_ids)
    start_offsets = self.motion_time_offsets[valid_env_ids]
    remaining_motion = torch.clamp(
      motion_lengths - start_offsets,
      min=1.0e-6,
    )
    attempted_duration = torch.minimum(
      remaining_motion,
      torch.full_like(remaining_motion, float(self._env.max_episode_length_s)),
    )
    elapsed_episode_time = episode_lengths[valid_mask].to(torch.float32) * float(
      self._env.step_dt
    )
    completion = torch.clamp(
      elapsed_episode_time / torch.clamp(attempted_duration, min=1.0e-6),
      min=0.0,
      max=1.0,
    )

    num_motions = self.motion_lib.num_motions()
    completion_sum = torch.zeros(num_motions, dtype=torch.float, device=self.device)
    completion_count = torch.zeros(num_motions, dtype=torch.float, device=self.device)
    completion_sum.scatter_add_(0, motion_ids, completion)
    completion_count.scatter_add_(0, motion_ids, torch.ones_like(completion))
    touched = completion_count > 0
    if not torch.any(touched):
      return

    batch_completion = completion_sum[touched] / completion_count[touched]
    alpha = float(self.cfg.motion_completion_alpha)
    alpha = min(max(alpha, 0.0), 1.0)
    first_attempt = self.motion_attempt_count[touched] == 0
    previous_completion = self.motion_completion_ema[touched]
    self.motion_completion_ema[touched] = torch.where(
      first_attempt,
      batch_completion,
      alpha * batch_completion + (1.0 - alpha) * previous_completion,
    )
    self.motion_attempt_count[touched] += completion_count[touched].to(torch.long)

  def _current_motion_sampling_probabilities(self) -> torch.Tensor | None:
    """Return the current per-motion sampling distribution."""
    if not self._uses_motion_library:
      return None
    assert self.motion_lib is not None
    num_motions = self.motion_lib.num_motions()
    if num_motions == 0:
      return None

    if self.cfg.sampling_mode == "adaptive" and self.motion_failed_count is not None:
      mix_ratio = float(min(max(self.cfg.adaptive_uniform_ratio, 0.0), 1.0))
      hard_motion_prob = self.motion_failed_count + 1.0 / float(num_motions)
      hard_motion_prob = hard_motion_prob / torch.clamp(
        hard_motion_prob.sum(), min=1.0e-8
      )
      probabilities = (
        1.0 - mix_ratio
      ) * hard_motion_prob + mix_ratio * self.motion_lib.motion_weights
    else:
      probabilities = self.motion_lib.motion_weights.clone()

    horizon_s = float(self.max_future_sampling_step_offset) * float(self._env.step_dt)
    valid_mask = (self.motion_lib.motion_lengths_s > (horizon_s + 1.0e-6)).to(
      probabilities.dtype
    )
    if torch.any(valid_mask > 0):
      probabilities = probabilities * valid_mask
    probabilities = probabilities / torch.clamp(probabilities.sum(), min=1.0e-8)
    return cap_sampling_probabilities(
      probabilities,
      max_ratio=self.cfg.adaptive_max_motion_sampling_ratio,
    )

  def write_motion_stats_csv(self, path: str | Path) -> bool:
    """Write compact per-motion training stats for post-training inspection."""
    if not self._uses_motion_library:
      return False
    assert self.motion_lib is not None
    if not hasattr(self, "motion_completion_ema"):
      return False

    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    probabilities = self._current_motion_sampling_probabilities()
    if probabilities is None:
      return False

    num_motions = self.motion_lib.num_motions()
    sampling_ratio = probabilities * float(num_motions)
    failure_ema = (
      self.motion_failed_count
      if self.motion_failed_count is not None
      else torch.zeros(num_motions, dtype=torch.float, device=self.device)
    )

    cwd = Path.cwd().resolve()
    with out_path.open("w", newline="", encoding="utf-8") as f:
      writer = csv.writer(f)
      writer.writerow(
        ["motion_file", "failure_ema", "completion_ema", "attempts", "sampling_ratio"]
      )
      for motion_file, fail, completion, attempts, ratio in zip(
        self.motion_lib.motion_files,
        failure_ema.detach().cpu().tolist(),
        self.motion_completion_ema.detach().cpu().tolist(),
        self.motion_attempt_count.detach().cpu().tolist(),
        sampling_ratio.detach().cpu().tolist(),
        strict=True,
      ):
        try:
          motion_file_str = str(Path(motion_file).resolve().relative_to(cwd))
        except ValueError:
          motion_file_str = str(motion_file)
        writer.writerow(
          [
            motion_file_str,
            f"{float(fail):.8g}",
            f"{float(completion):.8g}",
            int(attempts),
            f"{float(ratio):.8g}",
          ]
        )
    return True

  def _current_times_s(self) -> torch.Tensor:
    """Return the current reference time in seconds for each environment."""
    return (
      self.time_steps.to(torch.float32) * self._env.step_dt + self.motion_time_offsets
    )

  def _refresh_motion_frame(self) -> None:
    """Refresh the cached current frame when using the multi-clip motion library."""
    if not self._uses_motion_library:
      return
    assert self.motion_lib is not None
    self._current_motion_frame = self.motion_lib.calc_motion_frame(
      self.motion_ids,
      self._current_times_s(),
      anchor_body_index=self.motion_anchor_body_index,
    )

  def query_motion_frames(self, step_offsets: tuple[int, ...]) -> MotionFrameBatch:
    """Query future reference frames at given step offsets."""
    if len(step_offsets) == 0:
      raise ValueError("`step_offsets` must contain at least one entry.")

    if self._uses_motion_library:
      return self._query_motion_frames_library(step_offsets)
    return self._query_motion_frames_npz(step_offsets)

  def _batched_env_origins(self, num_steps: int) -> torch.Tensor:
    """Repeat environment origins for a flattened multi-step frame query."""
    return (
      self._env.scene.env_origins[:, None, :].expand(-1, num_steps, -1).reshape(-1, 3)
    )

  def _reshape_motion_frame_batch(
    self,
    *,
    num_steps: int,
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    body_pos_w: torch.Tensor,
    body_quat_w: torch.Tensor,
    body_lin_vel_w: torch.Tensor,
    body_ang_vel_w: torch.Tensor,
    anchor_pos_w: torch.Tensor,
    anchor_quat_w: torch.Tensor,
    anchor_lin_vel_w: torch.Tensor,
    anchor_ang_vel_w: torch.Tensor,
  ) -> MotionFrameBatch:
    """Pack flattened query results into the standard [env, step, ...] batch form."""
    num_bodies = len(self.cfg.body_names)
    return MotionFrameBatch(
      joint_pos=joint_pos.reshape(self.num_envs, num_steps, -1),
      joint_vel=joint_vel.reshape(self.num_envs, num_steps, -1),
      body_pos_w=body_pos_w.reshape(self.num_envs, num_steps, num_bodies, 3),
      body_quat_w=body_quat_w.reshape(self.num_envs, num_steps, num_bodies, 4),
      body_lin_vel_w=body_lin_vel_w.reshape(self.num_envs, num_steps, num_bodies, 3),
      body_ang_vel_w=body_ang_vel_w.reshape(self.num_envs, num_steps, num_bodies, 3),
      anchor_pos_w=anchor_pos_w.reshape(self.num_envs, num_steps, 3),
      anchor_quat_w=anchor_quat_w.reshape(self.num_envs, num_steps, 4),
      anchor_lin_vel_w=anchor_lin_vel_w.reshape(self.num_envs, num_steps, 3),
      anchor_ang_vel_w=anchor_ang_vel_w.reshape(self.num_envs, num_steps, 3),
    )

  def _query_motion_frames_npz(self, step_offsets: tuple[int, ...]) -> MotionFrameBatch:
    """Query future frames from a single loaded motion file."""
    assert self.motion is not None

    offsets = torch.tensor(step_offsets, dtype=torch.long, device=self.device)
    num_steps = int(offsets.shape[0])
    frame_ids = self.time_steps[:, None] + offsets[None, :]
    frame_ids = torch.clamp(frame_ids, min=0, max=self.motion.time_step_total - 1)
    flat_frame_ids = frame_ids.reshape(-1)
    origins = self._batched_env_origins(num_steps)

    body_pos = (
      self.motion.body_pos_w[flat_frame_ids][:, self.motion_body_indexes]
      + origins[:, None, :]
    )
    body_quat = self.motion.body_quat_w[flat_frame_ids][:, self.motion_body_indexes]
    body_lin_vel = self.motion.body_lin_vel_w[flat_frame_ids][
      :, self.motion_body_indexes
    ]
    body_ang_vel = self.motion.body_ang_vel_w[flat_frame_ids][
      :, self.motion_body_indexes
    ]

    anchor_pos = (
      self.motion.body_pos_w[flat_frame_ids, self.motion_anchor_body_index] + origins
    )
    anchor_quat = self.motion.body_quat_w[flat_frame_ids, self.motion_anchor_body_index]
    anchor_lin_vel = self.motion.body_lin_vel_w[
      flat_frame_ids, self.motion_anchor_body_index
    ]
    anchor_ang_vel = self.motion.body_ang_vel_w[
      flat_frame_ids, self.motion_anchor_body_index
    ]

    return self._reshape_motion_frame_batch(
      num_steps=num_steps,
      joint_pos=self.motion.joint_pos[flat_frame_ids],
      joint_vel=self.motion.joint_vel[flat_frame_ids],
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      body_lin_vel_w=body_lin_vel,
      body_ang_vel_w=body_ang_vel,
      anchor_pos_w=anchor_pos,
      anchor_quat_w=anchor_quat,
      anchor_lin_vel_w=anchor_lin_vel,
      anchor_ang_vel_w=anchor_ang_vel,
    )

  def _query_motion_frames_library(
    self, step_offsets: tuple[int, ...]
  ) -> MotionFrameBatch:
    """Query future frames from the multi-clip motion library."""
    assert self.motion_lib is not None

    offsets = torch.tensor(step_offsets, dtype=torch.float32, device=self.device)
    num_steps = int(offsets.shape[0])

    motion_ids = self.motion_ids[:, None].expand(-1, num_steps)
    query_times = (
      self._current_times_s()[:, None] + offsets[None, :] * self._env.step_dt
    )
    motion_lengths = self.motion_lib.get_motion_length(motion_ids.reshape(-1)).reshape(
      self.num_envs, num_steps
    )
    query_times = torch.clamp(query_times, min=0.0)
    query_times = torch.minimum(
      query_times, torch.clamp(motion_lengths - 1e-6, min=0.0)
    )

    flat_motion_ids = motion_ids.reshape(-1)
    flat_query_times = query_times.reshape(-1)
    flat_frames = self.motion_lib.calc_motion_frame(
      flat_motion_ids,
      flat_query_times,
      anchor_body_index=self.motion_anchor_body_index,
    )

    origins = self._batched_env_origins(num_steps)
    body_pos = flat_frames.body_pos_w[:, self.motion_body_indexes] + origins[:, None, :]
    body_quat = flat_frames.body_quat_w[:, self.motion_body_indexes]
    body_lin_vel = flat_frames.body_lin_vel_w[:, self.motion_body_indexes]
    body_ang_vel = flat_frames.body_ang_vel_w[:, self.motion_body_indexes]
    anchor_pos = flat_frames.body_pos_w[:, self.motion_anchor_body_index] + origins
    anchor_quat = flat_frames.body_quat_w[:, self.motion_anchor_body_index]
    anchor_lin_vel = flat_frames.body_lin_vel_w[:, self.motion_anchor_body_index]
    anchor_ang_vel = flat_frames.body_ang_vel_w[:, self.motion_anchor_body_index]

    return self._reshape_motion_frame_batch(
      num_steps=num_steps,
      joint_pos=flat_frames.joint_pos,
      joint_vel=flat_frames.joint_vel,
      body_pos_w=body_pos,
      body_quat_w=body_quat,
      body_lin_vel_w=body_lin_vel,
      body_ang_vel_w=body_ang_vel,
      anchor_pos_w=anchor_pos,
      anchor_quat_w=anchor_quat,
      anchor_lin_vel_w=anchor_lin_vel,
      anchor_ang_vel_w=anchor_ang_vel,
    )

  @property
  def command_representation_names(self) -> tuple[str, ...]:
    return ("default",)

  def has_command_representation(self, representation_name: str) -> bool:
    return representation_name in self.command_representation_names

  def get_command_representation(
    self, representation_name: str = "default"
  ) -> torch.Tensor:
    raise KeyError(
      f"{self.__class__.__name__} does not define command representation "
      f"{representation_name!r}. Available representations: "
      f"{self.command_representation_names}."
    )

  @property
  def command(self) -> torch.Tensor:
    return self.get_command_representation("default")

  @property
  def future_sampling_step_offsets(self) -> tuple[int, ...]:
    """Future offsets that must remain valid when sampling motion state."""
    return ()

  @property
  def max_future_sampling_step_offset(self) -> int:
    """Largest future step offset required by the active command representation."""
    offsets = self.future_sampling_step_offsets
    if len(offsets) == 0:
      return 0
    return max(int(offset) for offset in offsets)

  def motion_expired(self, step_offset: int = 0) -> torch.Tensor:
    """Return which environments would step past the end of the motion clip.

    Args:
      step_offset: Additional environment steps to project forward before checking
        whether the current clip would move beyond its final reference frame.
    """
    if step_offset < 0:
      raise ValueError(f"`step_offset` must be non-negative, got {step_offset}.")

    if self._uses_motion_library:
      assert self.motion_lib is not None
      query_times = self._current_times_s() + float(step_offset) * float(
        self._env.step_dt
      )
      motion_lengths = self.motion_lib.get_motion_length(self.motion_ids)
      return query_times > motion_lengths

    assert self.motion is not None
    return self.time_steps + step_offset >= self.motion.time_step_total

  def _reference_frame_tensor(self, name: str) -> torch.Tensor:
    """Return the current reference tensor from the active motion source."""
    if self._uses_motion_library:
      assert self._current_motion_frame is not None
      return getattr(self._current_motion_frame, name)
    assert self.motion is not None
    return getattr(self.motion, name)[self.time_steps]

  def _reference_body_tensor(self, name: str) -> torch.Tensor:
    """Return the tracked-body slice of the current reference tensor."""
    return self._reference_frame_tensor(name)[:, self.motion_body_indexes]

  def _reference_single_body_tensor(self, name: str, body_index: int) -> torch.Tensor:
    """Return the current reference tensor for one body index."""
    return self._reference_frame_tensor(name)[:, body_index]

  @property
  def joint_pos(self) -> torch.Tensor:
    return self._reference_frame_tensor("joint_pos")

  @property
  def joint_vel(self) -> torch.Tensor:
    return self._reference_frame_tensor("joint_vel")

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self._reference_body_tensor("body_pos_w")
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._reference_body_tensor("body_quat_w")

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._reference_body_tensor("body_lin_vel_w")

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._reference_body_tensor("body_ang_vel_w")

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self._reference_single_body_tensor("body_pos_w", self.motion_anchor_body_index)
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self._reference_single_body_tensor(
      "body_quat_w", self.motion_anchor_body_index
    )

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self._reference_single_body_tensor(
      "body_lin_vel_w", self.motion_anchor_body_index
    )

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self._reference_single_body_tensor(
      "body_ang_vel_w", self.motion_anchor_body_index
    )

  @property
  def root_pos_w(self) -> torch.Tensor:
    return (
      self._reference_single_body_tensor("body_pos_w", self.motion_root_body_index)
      + self._env.scene.env_origins
    )

  @property
  def root_quat_w(self) -> torch.Tensor:
    return self._reference_single_body_tensor(
      "body_quat_w", self.motion_root_body_index
    )

  @property
  def root_lin_vel_w(self) -> torch.Tensor:
    return self._reference_single_body_tensor(
      "body_lin_vel_w", self.motion_root_body_index
    )

  @property
  def root_ang_vel_w(self) -> torch.Tensor:
    return self._reference_single_body_tensor(
      "body_ang_vel_w", self.motion_root_body_index
    )

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

    self._metric_step_counts += 1
    for metric_name, metric_value in self.metrics.items():
      self._metric_episode_sums[metric_name] += metric_value

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    adaptive_sampling(self, env_ids)

  def _uniform_sampling(self, env_ids: torch.Tensor):
    uniform_sampling(self, env_ids)

  def _resample_command(self, env_ids: torch.Tensor):
    resample_command(self, env_ids)

  def _update_command(self):
    update_command(self)

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    debug_visualize_motion_command(self, visualizer)


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  """Shared YAHMP motion command configuration options."""

  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  root_body_name: str = ""
  motion_body_names: tuple[str, ...] | None = None
  entity_name: str
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.1, 0.1)
  adaptive_kernel_size: int = 3
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.8
  adaptive_max_motion_sampling_ratio: float = 200.0
  adaptive_alpha: float = 0.001
  motion_completion_alpha: float = 0.02
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  show_motion_load_progress: bool | None = None

  @dataclass
  class VizCfg:
    """Debug-visualization configuration for motion commands."""

    mode: Literal["ghost"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)

  def build(self, env: ManagerBasedRlEnv) -> MotionCommand:
    raise NotImplementedError(
      f"{self.__class__.__name__} must implement build() for a concrete motion command."
    )
