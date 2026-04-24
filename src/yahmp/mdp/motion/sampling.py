"""Sampling, reset, and per-step update helpers for YAHMP motion commands."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from mjlab.utils.lab_api.math import (
  quat_apply,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)

if TYPE_CHECKING:
  from .base import MotionCommand


def _max_future_offset_steps(command: MotionCommand) -> int:
  return command.max_future_sampling_step_offset


def _future_horizon_s(command: MotionCommand) -> float:
  return float(_max_future_offset_steps(command)) * float(command._env.step_dt)


def _episode_failed_mask(command: MotionCommand, env_ids: torch.Tensor) -> torch.Tensor:
  """Return resets that should count as actual failures for adaptive sampling."""
  return command._env.termination_manager.terminated[env_ids].clone()


def _sample_motion_ids_with_horizon(command: MotionCommand, n: int) -> torch.Tensor:
  """Sample motion ids while preferring clips that can satisfy future horizon."""
  assert command.motion_lib is not None
  horizon_s = _future_horizon_s(command)
  valid_mask = command.motion_lib.motion_lengths_s > (horizon_s + 1.0e-6)
  if torch.any(valid_mask):
    prob = command.motion_lib.motion_weights * valid_mask.to(
      command.motion_lib.motion_weights.dtype
    )
    prob = prob / torch.clamp(prob.sum(), min=1.0e-8)
    return torch.multinomial(prob, num_samples=n, replacement=True)
  return command.motion_lib.sample_motions(n)


def cap_sampling_probabilities(
  probabilities: torch.Tensor,
  *,
  max_ratio: float,
) -> torch.Tensor:
  """Cap a categorical distribution's top probability by ratio-to-uniform.

  ``max_ratio=200`` means no entry can have more than ``200 / N`` probability.
  The clipped probability mass is redistributed over entries that are still
  below the cap. Entries with zero probability stay zero, so validity masks are
  preserved.
  """
  if probabilities.numel() <= 1:
    return probabilities
  if max_ratio <= 0.0:
    return probabilities / torch.clamp(probabilities.sum(), min=1.0e-8)

  probabilities = probabilities / torch.clamp(probabilities.sum(), min=1.0e-8)
  positive = probabilities > 0.0
  num_positive = int(torch.count_nonzero(positive).item())
  if num_positive <= 1:
    return probabilities

  cap = float(max_ratio) / float(num_positive)
  if cap >= 1.0:
    return probabilities
  cap_tensor = torch.tensor(cap, dtype=probabilities.dtype, device=probabilities.device)
  capped = torch.zeros_like(probabilities)
  free = positive.clone()
  remaining_mass = torch.tensor(
    1.0, dtype=probabilities.dtype, device=probabilities.device
  )

  for _ in range(num_positive):
    base = probabilities[free]
    if base.numel() == 0:
      break
    candidate = base / torch.clamp(base.sum(), min=1.0e-8) * remaining_mass
    over_cap_local = candidate > cap_tensor
    if not torch.any(over_cap_local):
      capped[free] = candidate
      break

    free_indices = torch.nonzero(free, as_tuple=False).flatten()
    newly_capped = free_indices[over_cap_local]
    capped[newly_capped] = cap_tensor
    free[newly_capped] = False
    remaining_mass = 1.0 - capped[positive & ~free].sum()
    if remaining_mass <= 0.0:
      break

  if torch.any(free):
    base = probabilities[free]
    capped[free] = base / torch.clamp(base.sum(), min=1.0e-8) * remaining_mass

  capped[~positive] = 0.0
  return capped / torch.clamp(capped.sum(), min=1.0e-8)


def adaptive_sampling(command: MotionCommand, env_ids: torch.Tensor) -> None:
  """Resample environments using failure-weighted adaptive motion probabilities."""
  if command._uses_motion_library:
    assert command.motion_lib is not None
    assert command.motion_failed_count is not None
    assert command._current_motion_failed is not None
    assert command.phase_failed_count is not None
    assert command._current_phase_failed is not None

    command._current_motion_failed.zero_()
    command._current_phase_failed.zero_()

    episode_failed = _episode_failed_mask(command, env_ids)
    if torch.any(episode_failed):
      failed_env_ids = env_ids[episode_failed]
      failed_motion_ids = command.motion_ids[failed_env_ids]
      failed_times = command._current_times_s()[failed_env_ids]
      failed_lengths = command.motion_lib.get_motion_length(failed_motion_ids)
      failed_phase_bins = torch.clamp(
        (
          failed_times / torch.clamp(failed_lengths, min=1.0e-6) * command.bin_count
        ).long(),
        0,
        command.bin_count - 1,
      )
      command._current_motion_failed[:] = torch.bincount(
        failed_motion_ids, minlength=command.motion_lib.num_motions()
      ).to(dtype=torch.float32)
      phase_flat_idx = failed_motion_ids * command.bin_count + failed_phase_bins
      command._current_phase_failed[:] = (
        torch.bincount(
          phase_flat_idx,
          minlength=command.motion_lib.num_motions() * command.bin_count,
        )
        .to(dtype=torch.float32)
        .reshape(command.motion_lib.num_motions(), command.bin_count)
      )

    mix_ratio = float(min(max(command.cfg.adaptive_uniform_ratio, 0.0), 1.0))
    hard_motion_prob = command.motion_failed_count + 1.0 / float(
      command.motion_lib.num_motions()
    )
    hard_motion_prob = hard_motion_prob / hard_motion_prob.sum()
    motion_probabilities = (
      1.0 - mix_ratio
    ) * hard_motion_prob + mix_ratio * command.motion_lib.motion_weights
    horizon_s = _future_horizon_s(command)
    valid_mask = (command.motion_lib.motion_lengths_s > (horizon_s + 1.0e-6)).to(
      motion_probabilities.dtype
    )
    if torch.any(valid_mask > 0):
      motion_probabilities = motion_probabilities * valid_mask
    motion_probabilities = motion_probabilities / motion_probabilities.sum()
    motion_probabilities = cap_sampling_probabilities(
      motion_probabilities,
      max_ratio=command.cfg.adaptive_max_motion_sampling_ratio,
    )

    hard_phase_prob = command.phase_failed_count + 1.0 / float(command.bin_count)
    hard_phase_prob = torch.nn.functional.pad(
      hard_phase_prob.unsqueeze(1),
      (0, command.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    hard_phase_prob = torch.nn.functional.conv1d(
      hard_phase_prob, command.kernel.view(1, 1, -1)
    ).squeeze(1)
    hard_phase_prob = hard_phase_prob / torch.clamp(
      hard_phase_prob.sum(dim=1, keepdim=True), min=1.0e-8
    )

    uniform_phase_prob = torch.full_like(
      hard_phase_prob, 1.0 / float(command.bin_count)
    )
    phase_probabilities = (
      1.0 - mix_ratio
    ) * hard_phase_prob + mix_ratio * uniform_phase_prob
    phase_probabilities = phase_probabilities / torch.clamp(
      phase_probabilities.sum(dim=1, keepdim=True), min=1.0e-8
    )

    sampled_motion_ids = torch.multinomial(
      motion_probabilities, len(env_ids), replacement=True
    )
    sampled_phase_prob = phase_probabilities[sampled_motion_ids]
    sampled_bins = torch.multinomial(sampled_phase_prob, 1, replacement=True).squeeze(
      -1
    )
    sampled_phase = (
      sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=command.device)
    ) / command.bin_count
    sampled_lengths = command.motion_lib.get_motion_length(sampled_motion_ids)
    sampled_offsets = sampled_phase * sampled_lengths
    sampled_max_start = torch.clamp(sampled_lengths - horizon_s - 1.0e-6, min=0.0)
    sampled_offsets = torch.minimum(sampled_offsets, sampled_max_start)
    sampled_offsets = torch.minimum(
      sampled_offsets,
      torch.clamp(sampled_lengths - 1.0e-6, min=0.0),
    )

    command.motion_ids[env_ids] = sampled_motion_ids
    command.motion_time_offsets[env_ids] = sampled_offsets
    command.time_steps[env_ids] = 0
    command._refresh_motion_frame()

    motion_top_prob, motion_top_idx = motion_probabilities.max(dim=0)
    phase_top_prob, phase_top_idx = phase_probabilities[motion_top_idx].max(dim=0)
    if command.motion_lib.num_motions() > 1:
      motion_entropy = -(
        motion_probabilities * (motion_probabilities + 1.0e-12).log()
      ).sum() / math.log(command.motion_lib.num_motions())
    else:
      motion_entropy = torch.tensor(1.0, device=command.device)
    if command.bin_count > 1:
      phase_entropy = -(
        phase_probabilities * (phase_probabilities + 1.0e-12).log()
      ).sum(dim=1).mean() / math.log(command.bin_count)
    else:
      phase_entropy = torch.tensor(1.0, device=command.device)

    command.metrics["sampling_motion_entropy"][:] = motion_entropy
    command.metrics["sampling_motion_top1_prob"][:] = motion_top_prob
    command.metrics["sampling_motion_top1_idx"][:] = (
      motion_top_idx.float() / command.motion_lib.num_motions()
    )
    command.metrics["sampling_phase_entropy"][:] = phase_entropy
    command.metrics["sampling_phase_top1_prob"][:] = phase_top_prob
    command.metrics["sampling_phase_top1_bin"][:] = (
      phase_top_idx.float() / command.bin_count
    )
    return

  assert command.motion is not None
  assert command.bin_failed_count is not None
  assert command._current_bin_failed is not None
  episode_failed = _episode_failed_mask(command, env_ids)
  command._current_bin_failed.zero_()
  if torch.any(episode_failed):
    current_bin_index = torch.clamp(
      (command.time_steps * command.bin_count)
      // max(command.motion.time_step_total, 1),
      0,
      command.bin_count - 1,
    )
    fail_bins = current_bin_index[env_ids][episode_failed]
    command._current_bin_failed[:] = torch.bincount(
      fail_bins, minlength=command.bin_count
    )

  sampling_probabilities = (
    command.bin_failed_count
    + command.cfg.adaptive_uniform_ratio / float(command.bin_count)
  )
  sampling_probabilities = torch.nn.functional.pad(
    sampling_probabilities.unsqueeze(0).unsqueeze(0),
    (0, command.cfg.adaptive_kernel_size - 1),
    mode="replicate",
  )
  sampling_probabilities = torch.nn.functional.conv1d(
    sampling_probabilities, command.kernel.view(1, 1, -1)
  ).view(-1)
  sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

  sampled_bins = torch.multinomial(
    sampling_probabilities, len(env_ids), replacement=True
  )
  command.time_steps[env_ids] = (
    (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=command.device))
    / command.bin_count
    * (command.motion.time_step_total - 1)
  ).long()
  max_start_idx = max(
    command.motion.time_step_total - 1 - _max_future_offset_steps(command), 0
  )
  command.time_steps[env_ids] = torch.clamp(
    command.time_steps[env_ids], max=max_start_idx
  )

  H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
  H_norm = H / math.log(command.bin_count)
  pmax, imax = sampling_probabilities.max(dim=0)
  command.metrics["sampling_motion_entropy"][:] = 0.0
  command.metrics["sampling_motion_top1_prob"][:] = 0.0
  command.metrics["sampling_motion_top1_idx"][:] = 0.0
  command.metrics["sampling_phase_entropy"][:] = H_norm
  command.metrics["sampling_phase_top1_prob"][:] = pmax
  command.metrics["sampling_phase_top1_bin"][:] = imax.float() / command.bin_count


def uniform_sampling(command: MotionCommand, env_ids: torch.Tensor) -> None:
  """Resample environments uniformly over clips/phases within the valid horizon."""
  if command._uses_motion_library:
    assert command.motion_lib is not None
    sampled_motion_ids = _sample_motion_ids_with_horizon(command, len(env_ids))
    command.motion_ids[env_ids] = sampled_motion_ids
    sampled_lengths = command.motion_lib.get_motion_length(sampled_motion_ids)
    horizon_s = _future_horizon_s(command)
    sampled_max_start = torch.clamp(sampled_lengths - horizon_s - 1.0e-6, min=0.0)
    command.motion_time_offsets[env_ids] = (
      torch.rand(sampled_max_start.shape, device=command.device) * sampled_max_start
    )
    command.time_steps[env_ids] = 0
    command._refresh_motion_frame()

    motion_probabilities = command.motion_lib.motion_weights
    motion_top_prob, motion_top_idx = motion_probabilities.max(dim=0)
    if command.motion_lib.num_motions() > 1:
      motion_entropy = -(
        motion_probabilities * (motion_probabilities + 1.0e-12).log()
      ).sum() / math.log(command.motion_lib.num_motions())
    else:
      motion_entropy = torch.tensor(1.0, device=command.device)
    command.metrics["sampling_motion_entropy"][:] = motion_entropy
    command.metrics["sampling_motion_top1_prob"][:] = motion_top_prob
    command.metrics["sampling_motion_top1_idx"][:] = (
      motion_top_idx.float() / command.motion_lib.num_motions()
    )
    command.metrics["sampling_phase_entropy"][:] = 1.0
    command.metrics["sampling_phase_top1_prob"][:] = 1.0 / command.bin_count
    command.metrics["sampling_phase_top1_bin"][:] = 0.5
  else:
    assert command.motion is not None
    max_start_idx = max(
      command.motion.time_step_total - 1 - _max_future_offset_steps(command), 0
    )
    command.time_steps[env_ids] = torch.randint(
      0, max_start_idx + 1, (len(env_ids),), device=command.device
    )
    command.metrics["sampling_motion_entropy"][:] = 0.0
    command.metrics["sampling_motion_top1_prob"][:] = 0.0
    command.metrics["sampling_motion_top1_idx"][:] = 0.0
    command.metrics["sampling_phase_entropy"][:] = 1.0
    command.metrics["sampling_phase_top1_prob"][:] = 1.0 / command.bin_count
    command.metrics["sampling_phase_top1_bin"][:] = 0.5


def resample_command(command: MotionCommand, env_ids: torch.Tensor) -> None:
  """Resample command state without directly modifying simulation state."""
  if command.cfg.sampling_mode == "start":
    if command._uses_motion_library:
      assert command.motion_lib is not None
      command.motion_ids[env_ids] = _sample_motion_ids_with_horizon(
        command, len(env_ids)
      )
      command.motion_time_offsets[env_ids] = 0.0
    command.time_steps[env_ids] = 0
    if command._uses_motion_library:
      command._refresh_motion_frame()
  elif command.cfg.sampling_mode == "uniform":
    uniform_sampling(command, env_ids)
  else:
    assert command.cfg.sampling_mode == "adaptive"
    adaptive_sampling(command, env_ids)


def reset_robot_to_reference(command: MotionCommand, env_ids: torch.Tensor) -> None:
  """Write the sampled reference state into sim for reset environments."""
  root_pos = command.root_pos_w[env_ids].clone()
  root_ori = command.root_quat_w[env_ids].clone()
  root_lin_vel = command.root_lin_vel_w[env_ids].clone()
  root_ang_vel = command.root_ang_vel_w[env_ids].clone()

  range_list = [
    command.cfg.pose_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=command.device)
  rand_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=command.device
  )
  root_pos += rand_samples[:, 0:3]
  orientations_delta = quat_from_euler_xyz(
    rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
  )
  root_ori = quat_mul(orientations_delta, root_ori)

  range_list = [
    command.cfg.velocity_range.get(key, (0.0, 0.0))
    for key in ["x", "y", "z", "roll", "pitch", "yaw"]
  ]
  ranges = torch.tensor(range_list, device=command.device)
  rand_samples = sample_uniform(
    ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=command.device
  )
  root_lin_vel += rand_samples[:, :3]
  root_ang_vel += rand_samples[:, 3:]

  joint_pos = command.joint_pos[env_ids].clone()
  joint_vel = command.joint_vel[env_ids].clone()

  joint_pos += sample_uniform(
    lower=command.cfg.joint_position_range[0],
    upper=command.cfg.joint_position_range[1],
    size=joint_pos.shape,
    device=joint_pos.device,  # type: ignore[arg-type]
  )
  soft_joint_pos_limits = command.robot.data.soft_joint_pos_limits[env_ids]
  joint_pos = torch.clip(
    joint_pos, soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
  )
  command.robot.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)

  root_state = torch.cat(
    [
      root_pos,
      root_ori,
      root_lin_vel,
      root_ang_vel,
    ],
    dim=-1,
  )
  command.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
  command.robot.clear_state(env_ids=env_ids)


def update_command(command: MotionCommand) -> None:
  """Advance motion time and refresh metrics without teleporting the robot."""
  command.time_steps += 1

  if command._uses_motion_library:
    command._refresh_motion_frame()
  else:
    assert command.motion is not None
    command.time_steps.clamp_(max=command.motion.time_step_total - 1)

  anchor_pos_w_repeat = command.anchor_pos_w[:, None, :].repeat(
    1, len(command.cfg.body_names), 1
  )
  anchor_quat_w_repeat = command.anchor_quat_w[:, None, :].repeat(
    1, len(command.cfg.body_names), 1
  )
  robot_anchor_pos_w_repeat = command.robot_anchor_pos_w[:, None, :].repeat(
    1, len(command.cfg.body_names), 1
  )
  robot_anchor_quat_w_repeat = command.robot_anchor_quat_w[:, None, :].repeat(
    1, len(command.cfg.body_names), 1
  )

  delta_pos_w = robot_anchor_pos_w_repeat
  delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
  delta_ori_w = yaw_quat(
    quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
  )

  command.body_quat_relative_w = quat_mul(delta_ori_w, command.body_quat_w)
  command.body_pos_relative_w = delta_pos_w + quat_apply(
    delta_ori_w, command.body_pos_w - anchor_pos_w_repeat
  )

  if command.cfg.sampling_mode == "adaptive":
    if command._uses_motion_library:
      assert command.motion_failed_count is not None
      assert command._current_motion_failed is not None
      assert command.phase_failed_count is not None
      assert command._current_phase_failed is not None
      command.motion_failed_count = (
        command.cfg.adaptive_alpha * command._current_motion_failed
        + (1 - command.cfg.adaptive_alpha) * command.motion_failed_count
      )
      command.phase_failed_count = (
        command.cfg.adaptive_alpha * command._current_phase_failed
        + (1 - command.cfg.adaptive_alpha) * command.phase_failed_count
      )
      command._current_motion_failed.zero_()
      command._current_phase_failed.zero_()
    else:
      assert command.bin_failed_count is not None
      assert command._current_bin_failed is not None
      command.bin_failed_count = (
        command.cfg.adaptive_alpha * command._current_bin_failed
        + (1 - command.cfg.adaptive_alpha) * command.bin_failed_count
      )
      command._current_bin_failed.zero_()
