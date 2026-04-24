from __future__ import annotations

from typing import TYPE_CHECKING

import mujoco_warp as mjwarp
import torch
import warp as wp
from mjlab.actuator import (
  BuiltinMotorActuatorCfg,
  BuiltinPositionActuatorCfg,
  BuiltinVelocityActuatorCfg,
)
from mjlab.entity import Entity
from mjlab.utils.lab_api.math import quat_apply

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.viewer.debug_visualizer import DebugVisualizer


class apply_torque_limited_body_force:
  """Apply sustained body forces limited by current torque feasibility.

  This is a step-mode event with internal timers. Each active disturbance:

  1. Picks one target body from ``asset_cfg.body_ids`` for each environment.
  2. Builds a point Jacobian at the selected application point.
  3. Computes a torque baseline from inverse dynamics and, optionally, the current
     commanded joint torques.
  4. Samples a Cartesian force direction and scales it so that the resulting
     generalized torque stays within actuator effort limits.
  5. Holds that force for ``duration_s`` and then clears it for a sampled
     ``cooldown_s``.

  The force is always written as a wrench at the body's center of mass. When the
  application point is offset from the CoM, the equivalent moment ``r x f`` is added
  to the torque part of the wrench.
  """

  def __init__(self, cfg, env: ManagerBasedRlEnv):
    self._cfg = cfg
    self._env = env
    self._device = env.device
    self._warp_device = str(env.device)
    self._step_dt = float(env.step_dt)

    asset_cfg = cfg.params["asset_cfg"]
    self._asset: Entity = env.scene[asset_cfg.name]
    self._entity_data = self._asset.data
    self._model = self._entity_data.model
    self._data = self._entity_data.data

    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, list):
      self._body_ids = torch.tensor(body_ids, device=self._device, dtype=torch.long)
    elif isinstance(body_ids, torch.Tensor):
      self._body_ids = body_ids.to(device=self._device, dtype=torch.long)
    else:
      self._body_ids = torch.arange(
        self._asset.num_bodies, device=self._device, dtype=torch.long
      )
    if len(self._body_ids) == 0:
      raise ValueError("apply_torque_limited_body_force requires at least one body.")

    self._global_body_ids = self._asset.indexing.body_ids[self._body_ids]

    joint_names = tuple(cfg.params.get("joint_names", ()))
    if joint_names:
      joint_ids, _ = self._asset.find_joints(joint_names, preserve_order=True)
      self._joint_ids = torch.tensor(joint_ids, device=self._device, dtype=torch.long)
    else:
      self._joint_ids = torch.arange(
        self._asset.num_joints, device=self._device, dtype=torch.long
      )
    if len(self._joint_ids) == 0:
      raise ValueError(
        "apply_torque_limited_body_force requires at least one monitored joint."
      )
    self._joint_dof_ids = self._asset.indexing.joint_v_adr[self._joint_ids]

    self._joint_kp = torch.zeros(
      len(self._joint_ids), device=self._device, dtype=torch.float32
    )
    self._joint_kd = torch.zeros_like(self._joint_kp)
    self._joint_effort_limit = torch.full_like(self._joint_kp, torch.inf)
    self._build_joint_actuator_tables()

    duration_s = cfg.params["duration_s"]
    cooldown_s = cfg.params["cooldown_s"]
    self._validate_positive_range("duration_s", duration_s)
    self._validate_positive_range("cooldown_s", cooldown_s, allow_zero=True)

    feasible_force_fraction_range = cfg.params.get(
      "feasible_force_fraction_range", (0.0, 1.0)
    )
    low, high = feasible_force_fraction_range
    if low < 0.0 or high < low or high > 1.0:
      raise ValueError(
        "feasible_force_fraction_range must satisfy 0 <= low <= high <= 1, got "
        f"{feasible_force_fraction_range}."
      )
    self._feasible_force_fraction_range = (float(low), float(high))
    self._max_force_magnitude = float(cfg.params.get("max_force_magnitude", 75.0))
    if self._max_force_magnitude <= 0.0:
      raise ValueError(
        f"max_force_magnitude must be positive, got {self._max_force_magnitude}."
      )
    self._force_ramp_time_fraction = float(
      cfg.params.get("force_ramp_time_fraction", 0.0)
    )
    if self._force_ramp_time_fraction < 0.0 or self._force_ramp_time_fraction > 0.5:
      raise ValueError(
        "force_ramp_time_fraction must satisfy 0 <= value <= 0.5, got "
        f"{self._force_ramp_time_fraction}."
      )

    self._dirichlet_alpha = float(cfg.params.get("dirichlet_alpha", 1.0))
    if self._dirichlet_alpha <= 0.0:
      raise ValueError(
        f"dirichlet_alpha must be positive, got {self._dirichlet_alpha}."
      )
    self._dirichlet = torch.distributions.Dirichlet(
      torch.full((3,), self._dirichlet_alpha, device=self._device)
    )
    self._eps = float(cfg.params.get("eps", 1.0e-6))
    self._subtract_commanded_torque_margin = bool(
      cfg.params.get("subtract_commanded_torque_margin", True)
    )
    self._use_current_qvel_for_inverse_dynamics = bool(
      cfg.params.get("use_current_qvel_for_inverse_dynamics", False)
    )
    self._randomize_application_point = bool(
      cfg.params.get("randomize_application_point", False)
    )
    self._randomize_body = bool(cfg.params.get("randomize_body", True))
    self._debug_force_vis_enabled = bool(
      cfg.params.get("debug_force_vis_enabled", True)
    )
    self._debug_force_vis_scale = float(cfg.params.get("debug_force_vis_scale", 0.015))
    if self._debug_force_vis_scale < 0.0:
      raise ValueError(
        "debug_force_vis_scale must be non-negative, got "
        f"{self._debug_force_vis_scale}."
      )
    self._debug_force_vis_width = float(cfg.params.get("debug_force_vis_width", 0.01))
    if self._debug_force_vis_width < 0.0:
      raise ValueError(
        "debug_force_vis_width must be non-negative, got "
        f"{self._debug_force_vis_width}."
      )

    body_point_offset = cfg.params.get("body_point_offset", None)
    self._body_point_offset = (
      torch.tensor(body_point_offset, device=self._device, dtype=torch.float32)
      if body_point_offset is not None
      else None
    )
    self._application_point_delta_range = cfg.params.get(
      "application_point_delta_range", None
    )

    self._active = torch.zeros(self.num_envs, device=self._device, dtype=torch.bool)
    self._time_remaining = torch.zeros(self.num_envs, device=self._device)
    self._cooldown_time_left = torch.zeros(self.num_envs, device=self._device)
    self._active_forces = torch.zeros(
      (self.num_envs, len(self._body_ids), 3), device=self._device
    )
    self._active_torques = torch.zeros_like(self._active_forces)
    self._peak_forces = torch.zeros_like(self._active_forces)
    self._peak_torques = torch.zeros_like(self._active_forces)
    self._active_body_slots = torch.full(
      (self.num_envs,), -1, device=self._device, dtype=torch.long
    )
    self._active_offset_local = torch.zeros(
      (self.num_envs, 3), device=self._device, dtype=torch.float32
    )
    self._active_duration_total = torch.zeros(
      self.num_envs, device=self._device, dtype=torch.float32
    )
    self._active_elapsed_time = torch.zeros(
      self.num_envs, device=self._device, dtype=torch.float32
    )
    self._episode_force_mag_sum = torch.zeros(
      self.num_envs, device=self._device, dtype=torch.float32
    )
    self._episode_force_sample_count = torch.zeros(
      self.num_envs, device=self._device, dtype=torch.long
    )
    self._episode_force_mag_max = torch.zeros(
      self.num_envs, device=self._device, dtype=torch.float32
    )

    self._jacobian_full = torch.zeros(
      (self.num_envs, 3, self._model.nv), device=self._device, dtype=torch.float32
    )
    self._jacobian_full_wp = wp.from_torch(self._jacobian_full)
    self._jacobian_body_ids = torch.zeros(
      self.num_envs, device=self._device, dtype=torch.int32
    )
    self._jacobian_body_ids_wp = wp.from_torch(self._jacobian_body_ids, dtype=wp.int32)
    self._jacobian_points = torch.zeros(
      (self.num_envs, 3), device=self._device, dtype=torch.float32
    )
    self._jacobian_points_wp = wp.from_torch(self._jacobian_points, dtype=wp.vec3)

    self._sample_cooldown(slice(None))

  @property
  def num_envs(self) -> int:
    return self._env.num_envs

  def set_feasible_force_fraction_range(self, low: float, high: float) -> None:
    """Update the feasible-force fraction range used for newly sampled disturbances."""
    if low < 0.0 or high < low or high > 1.0:
      raise ValueError(
        "feasible_force_fraction_range must satisfy 0 <= low <= high <= 1, got "
        f"({low}, {high})."
      )
    self._feasible_force_fraction_range = (float(low), float(high))
    self._cfg.params["feasible_force_fraction_range"] = (
      self._feasible_force_fraction_range
    )

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    if env_ids is None:
      env_ids = slice(None)
    self._active[env_ids] = False
    self._time_remaining[env_ids] = 0.0
    self._active_forces[env_ids] = 0.0
    self._active_torques[env_ids] = 0.0
    self._peak_forces[env_ids] = 0.0
    self._peak_torques[env_ids] = 0.0
    self._active_body_slots[env_ids] = -1
    self._active_offset_local[env_ids] = 0.0
    self._active_duration_total[env_ids] = 0.0
    self._active_elapsed_time[env_ids] = 0.0
    self._episode_force_mag_sum[env_ids] = 0.0
    self._episode_force_sample_count[env_ids] = 0
    self._episode_force_mag_max[env_ids] = 0.0
    self._asset.write_external_wrench_to_sim(
      self._active_forces[env_ids],
      self._active_torques[env_ids],
      env_ids=env_ids,
      body_ids=self._body_ids.tolist(),
    )
    self._sample_cooldown(env_ids)

  def __call__(
    self, env: ManagerBasedRlEnv, env_ids: torch.Tensor | None, **kwargs
  ) -> None:
    del env, env_ids, kwargs

    dt = self._step_dt

    active_ids = self._active.nonzero(as_tuple=False).flatten()
    if len(active_ids) > 0:
      self._active_elapsed_time[active_ids] += dt
      remaining = (
        self._active_duration_total[active_ids] - self._active_elapsed_time[active_ids]
      )
      self._time_remaining[active_ids] = remaining.clamp_min(0.0)
      still_active = active_ids[remaining > 0.0]
      expired = active_ids[remaining <= 0.0]
      if len(still_active) > 0:
        self._write_active_wrench(still_active)
      if len(expired) > 0:
        self._active[expired] = False
        self._time_remaining[expired] = 0.0
        self._active_forces[expired] = 0.0
        self._active_torques[expired] = 0.0
        self._peak_forces[expired] = 0.0
        self._peak_torques[expired] = 0.0
        self._active_body_slots[expired] = -1
        self._active_offset_local[expired] = 0.0
        self._active_duration_total[expired] = 0.0
        self._active_elapsed_time[expired] = 0.0
        self._asset.write_external_wrench_to_sim(
          self._active_forces[expired],
          self._active_torques[expired],
          env_ids=expired,
          body_ids=self._body_ids.tolist(),
        )
        self._sample_cooldown(expired)

    inactive = (~self._active).nonzero(as_tuple=False).flatten()
    if len(inactive) == 0:
      return

    self._cooldown_time_left[inactive] -= dt
    trigger_ids = inactive[self._cooldown_time_left[inactive] <= 0.0]
    if len(trigger_ids) == 0:
      return

    body_slots, body_ids, point_w, offset_local, moment_arm_w = (
      self._sample_application_points(trigger_ids)
    )
    baseline_torque = self._compute_baseline_torque(trigger_ids)
    jacobian = self._compute_point_jacobian(trigger_ids, body_ids, point_w)

    force = self._sample_force_vector(jacobian, baseline_torque)
    torque = torch.cross(moment_arm_w, force, dim=-1)
    force_mag = torch.linalg.vector_norm(force, dim=-1)

    forces = torch.zeros(
      (len(trigger_ids), len(self._body_ids), 3), device=self._device
    )
    torques = torch.zeros_like(forces)
    row_ids = torch.arange(len(trigger_ids), device=self._device)
    forces[row_ids, body_slots] = force
    torques[row_ids, body_slots] = torque

    duration = self._sample_range_tensor(
      self._cfg.params["duration_s"], len(trigger_ids)
    )

    self._peak_forces[trigger_ids] = forces
    self._peak_torques[trigger_ids] = torques
    self._active_body_slots[trigger_ids] = body_slots
    self._active_offset_local[trigger_ids] = offset_local
    self._active_duration_total[trigger_ids] = duration
    self._active_elapsed_time[trigger_ids] = 0.0
    self._episode_force_mag_sum[trigger_ids] += force_mag
    self._episode_force_sample_count[trigger_ids] += 1
    self._episode_force_mag_max[trigger_ids] = torch.maximum(
      self._episode_force_mag_max[trigger_ids], force_mag
    )

    self._active[trigger_ids] = True
    self._time_remaining[trigger_ids] = duration
    self._write_active_wrench(trigger_ids)

  def get_episode_force_magnitude_stats(
    self, env_ids: torch.Tensor | slice | None = None
  ) -> dict[str, torch.Tensor]:
    """Return reset-time force magnitude summaries for the selected envs."""
    if env_ids is None:
      env_ids = slice(None)

    sample_count = self._episode_force_sample_count[env_ids].to(torch.float32)
    active_mask = sample_count > 0
    total_sample_count = sample_count.sum()
    zero = torch.zeros((), device=self._device, dtype=torch.float32)

    if bool(active_mask.any().item()):
      magnitude_mean = self._episode_force_mag_sum[env_ids].sum() / total_sample_count
      magnitude_max = self._episode_force_mag_max[env_ids][active_mask].max()
    else:
      magnitude_mean = zero
      magnitude_max = zero

    return {
      "magnitude_mean": magnitude_mean,
      "magnitude_max": magnitude_max,
      "episode_sample_count_mean": sample_count.mean(),
      "episode_with_force_frac": active_mask.to(torch.float32).mean(),
    }

  def debug_vis(self, visualizer: DebugVisualizer) -> None:
    """Draw the active external body force as an arrow in the viewer."""
    if not self._debug_force_vis_enabled:
      return

    env_indices = visualizer.get_env_indices(self.num_envs)
    if not env_indices:
      return

    color = (1.0, 0.55, 0.0, 1.0)

    for env_id in env_indices:
      if not bool(self._active[env_id].item()):
        continue

      body_slot = int(self._active_body_slots[env_id].item())
      if body_slot < 0:
        continue

      force = self._active_forces[env_id, body_slot]
      force_norm = float(torch.linalg.vector_norm(force).item())
      if force_norm <= self._eps:
        continue

      body_id = int(self._body_ids[body_slot].item())
      body_pos_w = self._entity_data.body_link_pos_w[env_id, body_id]
      body_quat_w = self._entity_data.body_link_quat_w[env_id, body_id]
      offset_local = self._active_offset_local[env_id]
      offset_w = quat_apply(body_quat_w.unsqueeze(0), offset_local.unsqueeze(0))[0]
      point_w = body_pos_w + offset_w
      arrow_vec = force * self._debug_force_vis_scale
      start_w = point_w
      end_w = start_w + arrow_vec

      visualizer.add_arrow(
        start_w,
        end_w,
        color=color,
        width=self._debug_force_vis_width,
        label=f"push_force_{env_id}",
      )

  def _compute_force_ramp_scale(self, env_ids: torch.Tensor) -> torch.Tensor:
    if self._force_ramp_time_fraction <= 0.0:
      return torch.ones(len(env_ids), device=self._device, dtype=torch.float32)

    total = self._active_duration_total[env_ids].clamp_min(self._eps)
    phase = ((self._active_elapsed_time[env_ids] + 0.5 * self._step_dt) / total).clamp(
      0.0, 1.0
    )
    ramp = self._force_ramp_time_fraction
    ramp_up = (phase / ramp).clamp(max=1.0)
    ramp_down = ((1.0 - phase) / ramp).clamp(max=1.0)
    return torch.minimum(ramp_up, ramp_down).clamp(min=0.0)

  def _write_active_wrench(self, env_ids: torch.Tensor) -> None:
    if len(env_ids) == 0:
      return
    scale = self._compute_force_ramp_scale(env_ids).view(-1, 1, 1)
    forces = self._peak_forces[env_ids] * scale
    torques = self._peak_torques[env_ids] * scale
    self._active_forces[env_ids] = forces
    self._active_torques[env_ids] = torques
    self._asset.write_external_wrench_to_sim(
      forces,
      torques,
      env_ids=env_ids,
      body_ids=self._body_ids.tolist(),
    )

  def _build_joint_actuator_tables(self) -> None:
    joint_to_slot = {
      joint_id.item(): idx for idx, joint_id in enumerate(self._joint_ids)
    }
    for actuator in self._asset.actuators:
      base_cfg = getattr(actuator.cfg, "base_cfg", actuator.cfg)
      if isinstance(base_cfg, BuiltinPositionActuatorCfg):
        kp = float(base_cfg.stiffness)
        kd = float(base_cfg.damping)
        effort_limit = (
          float(base_cfg.effort_limit)
          if base_cfg.effort_limit is not None
          else torch.inf
        )
      elif isinstance(base_cfg, BuiltinVelocityActuatorCfg):
        kp = 0.0
        kd = float(base_cfg.damping)
        effort_limit = (
          float(base_cfg.effort_limit)
          if base_cfg.effort_limit is not None
          else torch.inf
        )
      elif isinstance(base_cfg, BuiltinMotorActuatorCfg):
        kp = 0.0
        kd = 0.0
        effort_limit = float(base_cfg.effort_limit)
      else:
        continue

      for target_id in actuator.target_ids.tolist():
        slot = joint_to_slot.get(target_id)
        if slot is None:
          continue
        self._joint_kp[slot] = kp
        self._joint_kd[slot] = kd
        self._joint_effort_limit[slot] = effort_limit

  def _compute_baseline_torque(self, env_ids: torch.Tensor) -> torch.Tensor:
    joint_dof_ids = self._joint_dof_ids
    with wp.ScopedDevice(self._warp_device):
      if self._use_current_qvel_for_inverse_dynamics:
        mjwarp.rne(self._model, self._data, flg_acc=False)
        tau_id = self._data.qfrc_bias[env_ids][:, joint_dof_ids].clone()
      else:
        original_qvel = self._data.qvel[env_ids].clone()
        self._data.qvel[env_ids] = 0.0
        mjwarp.rne(self._model, self._data, flg_acc=False)
        tau_id = self._data.qfrc_bias[env_ids][:, joint_dof_ids].clone()
        self._data.qvel[env_ids] = original_qvel
        mjwarp.rne(self._model, self._data, flg_acc=False)

    if not self._subtract_commanded_torque_margin:
      return tau_id

    joint_pos = self._entity_data.joint_pos[env_ids][:, self._joint_ids]
    joint_vel = self._entity_data.joint_vel[env_ids][:, self._joint_ids]
    pos_target = self._entity_data.joint_pos_target[env_ids][:, self._joint_ids]
    vel_target = self._entity_data.joint_vel_target[env_ids][:, self._joint_ids]
    effort_target = self._entity_data.joint_effort_target[env_ids][:, self._joint_ids]

    tau_cmd = self._joint_kp.unsqueeze(0) * (pos_target - joint_pos)
    tau_cmd += self._joint_kd.unsqueeze(0) * (vel_target - joint_vel)
    tau_cmd += effort_target

    finite_limit = torch.isfinite(self._joint_effort_limit)
    if finite_limit.any():
      limit = self._joint_effort_limit.unsqueeze(0).expand_as(tau_cmd)
      tau_cmd = torch.where(
        finite_limit.unsqueeze(0),
        tau_cmd.clamp(-limit, limit),
        tau_cmd,
      )

    return tau_id + tau_cmd

  def _compute_point_jacobian(
    self,
    env_ids: torch.Tensor,
    body_ids: torch.Tensor,
    point_w: torch.Tensor,
  ) -> torch.Tensor:
    self._jacobian_body_ids[env_ids] = body_ids.to(torch.int32)
    self._jacobian_points[env_ids] = point_w
    with wp.ScopedDevice(self._warp_device):
      mjwarp.jac(
        self._model,
        self._data,
        self._jacobian_full_wp,
        None,
        self._jacobian_points_wp,
        self._jacobian_body_ids_wp,
      )
    return self._jacobian_full[env_ids][:, :, self._joint_dof_ids]

  def _sample_force_vector(
    self,
    jacobian: torch.Tensor,
    baseline_torque: torch.Tensor,
  ) -> torch.Tensor:
    pos_capacity = torch.stack(
      [
        self._max_scale_for_direction(jacobian[:, axis, :], baseline_torque)
        for axis in range(3)
      ],
      dim=-1,
    )
    neg_capacity = torch.stack(
      [
        self._max_scale_for_direction(-jacobian[:, axis, :], baseline_torque)
        for axis in range(3)
      ],
      dim=-1,
    )

    sign = torch.zeros_like(pos_capacity)
    both = (pos_capacity > self._eps) & (neg_capacity > self._eps)
    random_sign = torch.where(
      torch.rand_like(pos_capacity) < 0.5,
      -torch.ones_like(pos_capacity),
      torch.ones_like(pos_capacity),
    )
    sign = torch.where(both, random_sign, sign)
    sign = torch.where((pos_capacity > self._eps) & ~both, torch.ones_like(sign), sign)
    sign = torch.where((neg_capacity > self._eps) & ~both, -torch.ones_like(sign), sign)

    axis_capacity = torch.where(sign > 0.0, pos_capacity, neg_capacity)
    active_axis = axis_capacity > self._eps
    weights = self._dirichlet.sample((jacobian.shape[0],))
    weights = weights * active_axis.float()
    weights_sum = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(
      weights_sum > self._eps,
      weights / weights_sum.clamp_min(self._eps),
      torch.zeros_like(weights),
    )

    direction = sign * weights * axis_capacity
    direction_scale = self._max_scale_for_direction_vector(
      direction, jacobian, baseline_torque
    )
    feasible_force_fraction = self._sample_range_tensor(
      self._feasible_force_fraction_range, jacobian.shape[0]
    )
    force = direction * (direction_scale * feasible_force_fraction).unsqueeze(-1)
    force_norm = torch.linalg.vector_norm(force, dim=-1, keepdim=True)
    cap_scale = (self._max_force_magnitude / force_norm.clamp_min(self._eps)).clamp(
      max=1.0
    )
    return force * cap_scale

  def _max_scale_for_direction(
    self,
    coeff: torch.Tensor,
    baseline_torque: torch.Tensor,
  ) -> torch.Tensor:
    effort_limit = self._joint_effort_limit.unsqueeze(0).expand_as(coeff)
    max_scale = torch.full_like(coeff, torch.inf)

    positive = coeff > self._eps
    negative = coeff < -self._eps

    max_scale = torch.where(
      positive,
      (effort_limit - baseline_torque) / coeff.clamp_min(self._eps),
      max_scale,
    )
    max_scale = torch.where(
      negative,
      (-effort_limit - baseline_torque) / coeff.clamp_max(-self._eps),
      max_scale,
    )

    active_mask = torch.abs(coeff) > self._eps
    baseline_feasible = torch.all(
      (~active_mask) | (torch.abs(baseline_torque) <= effort_limit + self._eps),
      dim=-1,
    )
    has_effect = torch.any(active_mask, dim=-1)
    scale = max_scale.min(dim=-1).values.clamp_min(0.0)
    scale = torch.where(baseline_feasible & has_effect, scale, torch.zeros_like(scale))
    return scale

  def _max_scale_for_direction_vector(
    self,
    direction: torch.Tensor,
    jacobian: torch.Tensor,
    baseline_torque: torch.Tensor,
  ) -> torch.Tensor:
    coeff = torch.bmm(jacobian.transpose(1, 2), direction.unsqueeze(-1)).squeeze(-1)
    return self._max_scale_for_direction(coeff, baseline_torque)

  def _sample_application_points(
    self, env_ids: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if self._randomize_body:
      body_slots = torch.randint(
        len(self._body_ids), (len(env_ids),), device=self._device
      )
    else:
      body_slots = torch.zeros(len(env_ids), device=self._device, dtype=torch.long)

    row_ids = torch.arange(len(env_ids), device=self._device)
    body_link_pos_all = self._entity_data.body_link_pos_w[env_ids][:, self._body_ids]
    body_link_quat_all = self._entity_data.body_link_quat_w[env_ids][:, self._body_ids]
    body_com_pos_all = self._entity_data.body_com_pos_w[env_ids][:, self._body_ids]
    body_link_pos = body_link_pos_all[row_ids, body_slots]
    body_link_quat = body_link_quat_all[row_ids, body_slots]
    body_com_pos = body_com_pos_all[row_ids, body_slots]

    offset_local = self._sample_body_point_offset(len(env_ids))
    if offset_local is None:
      offset_local = torch.zeros((len(env_ids), 3), device=self._device)
      offset_from_link_w = offset_local
    else:
      offset_from_link_w = quat_apply(body_link_quat, offset_local)

    point_w = body_link_pos + offset_from_link_w
    moment_arm_w = point_w - body_com_pos
    body_ids_global = self._global_body_ids[body_slots]
    return body_slots, body_ids_global, point_w, offset_local, moment_arm_w

  def _sample_body_point_offset(self, batch: int) -> torch.Tensor | None:
    center = self._body_point_offset
    if (
      self._randomize_application_point
      and self._application_point_delta_range is not None
    ):
      ranges = torch.tensor(
        self._application_point_delta_range,
        device=self._device,
        dtype=torch.float32,
      )
      jitter = self._sample_uniform_from_ranges(ranges, batch)
      if center is None:
        return jitter
      return center.unsqueeze(0) + jitter
    if center is not None:
      return center.unsqueeze(0).expand(batch, -1)
    return None

  def _sample_cooldown(self, env_ids: torch.Tensor | slice) -> None:
    num_envs = self.num_envs if isinstance(env_ids, slice) else len(env_ids)
    self._cooldown_time_left[env_ids] = self._sample_range_tensor(
      self._cfg.params["cooldown_s"], num_envs
    )

  def _sample_range_tensor(
    self,
    value_range: tuple[float, float],
    batch: int,
  ) -> torch.Tensor:
    low, high = value_range
    return torch.rand(batch, device=self._device) * (high - low) + low

  def _sample_uniform_from_ranges(
    self,
    ranges: torch.Tensor,
    batch: int,
  ) -> torch.Tensor:
    return torch.rand((batch, ranges.shape[0]), device=self._device) * (
      ranges[:, 1] - ranges[:, 0]
    ).unsqueeze(0) + ranges[:, 0].unsqueeze(0)

  @staticmethod
  def _validate_positive_range(
    name: str,
    value_range: tuple[float, float],
    allow_zero: bool = False,
  ) -> None:
    low, high = value_range
    if high < low or low < 0.0 or (not allow_zero and low <= 0.0):
      expected = "0 <= low <= high" if allow_zero else "0 < low <= high"
      raise ValueError(f"{name} must satisfy {expected}, got {value_range}.")
