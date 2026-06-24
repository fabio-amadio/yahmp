"""Headless evaluation of a locomotion checkpoint on random velocity commands.

Commands are sampled by the environment's training command term after applying
the cumulative final velocity-curriculum stage. Observation corruption and all
non-reset events (pushes and domain randomization) are disabled.

Outputs:
  - rollout.npz: dense per-step tensors with shape [step, env, ...]
  - per_env_summary.csv: tracking and actuation aggregates per environment
  - metadata.json: configuration, column names, shapes, and global summary

Example:
  uv run python scripts/evaluate_random_commands.py \
      Mjlab-YAHMP-Locomotion-Unitree-G1 \
      --checkpoint-file /path/to/model_4999.pt \
      --imitation-checkpoint-file assets/models/multi_task/imitation_rvq_model.pt \
      --rvq-num-active-quantizers 5 \
      --num-envs 64
"""

from __future__ import annotations

import csv
import json
import math
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import mjlab
import numpy as np
import torch
import tyro
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner, RslRlVecEnvWrapper
from mjlab.tasks.registry import (
    _REGISTRY,
    list_tasks,
    load_env_cfg,
    load_rl_cfg,
    load_runner_cls,
)
from mjlab.utils.torch import configure_torch_backends


@dataclass(frozen=True)
class Config:
    checkpoint_file: str
    """Path to the trained locomotion checkpoint."""

    imitation_checkpoint_file: str | None = None
    """Frozen imitation/RVQ checkpoint required by the YAHMP locomotion runner."""

    rvq_num_active_quantizers: int | None = None
    """Number of active RVQ quantizers; must match training."""

    output_dir: str | None = None
    """Output directory. Defaults to outputs/random_command_eval/<timestamp>."""

    num_envs: int = 64
    episode_length_s: float | None = None
    """Rollout duration. None uses the training environment episode length."""

    seed: int = 1
    device: str | None = None
    fall_angle_deg: float = 70.0
    compress: bool = True


def _find_twist_term_name(active_terms: list[str]) -> str:
    for candidate in ("twist", "base_velocity", "velocity"):
        if candidate in active_terms:
            return candidate
    for name in active_terms:
        if "twist" in name.lower() or "velocity" in name.lower():
            return name
    raise RuntimeError(
        f"Could not locate a twist/velocity command term among {active_terms!r}."
    )


def _apply_last_velocity_stage(env_cfg: Any, command_name: str) -> list[dict[str, Any]]:
    curriculum_cfg = env_cfg.curriculum.get("command_vel")
    if curriculum_cfg is None:
        return []
    if curriculum_cfg.params.get("command_name") != command_name:
        return []

    stages = list(curriculum_cfg.params.get("velocity_stages", ()))
    command_cfg = env_cfg.commands[command_name]
    for stage in sorted(stages, key=lambda item: int(item["step"])):
        for field in ("lin_vel_x", "lin_vel_y", "ang_vel_z"):
            value = stage.get(field)
            if value is not None:
                setattr(command_cfg.ranges, field, tuple(value))
    return stages


def _configure_evaluation_env(
    env_cfg: Any,
    *,
    num_envs: int,
    episode_length_s: float,
    seed: int,
) -> tuple[str, list[dict[str, Any]]]:
    env_cfg.seed = seed
    env_cfg.scene.num_envs = num_envs
    env_cfg.episode_length_s = episode_length_s

    for group in env_cfg.observations.values():
        if group is not None:
            group.enable_corruption = False

    # Keep deterministic state-reset events; remove pushes and startup DR.
    env_cfg.events = {
        name: event for name, event in env_cfg.events.items() if event.mode == "reset"
    }
    reset_base = env_cfg.events.get("reset_base")
    if reset_base is not None:
        reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        reset_base.params["velocity_range"] = {}
    reset_joints = env_cfg.events.get("reset_robot_joints")
    if reset_joints is not None:
        reset_joints.params["position_range"] = (0.0, 0.0)
        reset_joints.params["velocity_range"] = (0.0, 0.0)

    command_name = _find_twist_term_name(list(env_cfg.commands))
    stages = _apply_last_velocity_stage(env_cfg, command_name)
    command_cfg = env_cfg.commands[command_name]
    command_cfg.debug_vis = False
    if hasattr(command_cfg, "init_velocity_prob"):
        command_cfg.init_velocity_prob = 0.0

    # A fixed loop controls episode duration. Falls are detected and masked
    # locally so mjlab cannot auto-reset and overwrite the terminal state.
    env_cfg.curriculum = {}
    env_cfg.terminations = {}
    return command_name, stages


def _to_numpy(tensor: torch.Tensor) -> np.ndarray:
    return tensor.detach().cpu().numpy()


def _float_buffer(num_steps: int, num_envs: int, *tail: int) -> np.ndarray:
    return np.empty((num_steps, num_envs, *tail), dtype=np.float32)


def _safe_mean(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _safe_rmse(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values)))) if values.size else math.nan


def _summarize_env(
    env_id: int,
    arrays: dict[str, np.ndarray],
    step_dt: float,
    first_fall_step: np.ndarray,
) -> dict[str, Any]:
    mask = arrays["valid"][:, env_id]
    error = arrays["velocity_error_b"][:, env_id][mask]
    torque = arrays["actuator_torque"][:, env_id][mask]
    action = arrays["action"][:, env_id][mask]
    processed = arrays["processed_action"][:, env_id][mask]

    row: dict[str, Any] = {
        "env_id": env_id,
        "valid_steps": int(mask.sum()),
        "tracked_duration_s": float(mask.sum()) * step_dt,
        "fell": bool(first_fall_step[env_id] >= 0),
        "fall_time_s": (
            float(first_fall_step[env_id] + 1) * step_dt
            if first_fall_step[env_id] >= 0
            else math.nan
        ),
        "rmse_vx_mps": _safe_rmse(error[:, 0]),
        "rmse_vy_mps": _safe_rmse(error[:, 1]),
        "rmse_wz_radps": _safe_rmse(error[:, 2]),
        "mean_linear_error_norm_mps": _safe_mean(np.linalg.norm(error[:, :2], axis=1)),
        "mean_abs_yaw_error_radps": _safe_mean(np.abs(error[:, 2])),
        "mean_abs_actuator_torque": _safe_mean(np.abs(torque)),
        "max_abs_actuator_torque": (
            float(np.max(np.abs(torque))) if torque.size else math.nan
        ),
        "mean_abs_action": _safe_mean(np.abs(action)),
        "mean_abs_processed_action": _safe_mean(np.abs(processed)),
        "standing_fraction": _safe_mean(
            arrays["is_standing_command"][:, env_id][mask].astype(np.float32)
        ),
    }
    return row


def _aggregate_summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metric_names = (
        "tracked_duration_s",
        "rmse_vx_mps",
        "rmse_vy_mps",
        "rmse_wz_radps",
        "mean_linear_error_norm_mps",
        "mean_abs_yaw_error_radps",
        "mean_abs_actuator_torque",
        "max_abs_actuator_torque",
        "mean_abs_action",
        "mean_abs_processed_action",
        "standing_fraction",
    )
    summary: dict[str, Any] = {
        "num_envs": len(rows),
        "num_falls": sum(int(row["fell"]) for row in rows),
        "fall_rate": (
            sum(int(row["fell"]) for row in rows) / len(rows) if rows else math.nan
        ),
    }
    for name in metric_names:
        values = np.asarray([float(row[name]) for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        summary[name + "_mean"] = _safe_mean(finite)
    return summary


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_jsonable(item) for item in value]
    return value


def main() -> None:
    configure_torch_backends()

    all_tasks = list_tasks()
    chosen_task, remaining_args = tyro.cli(
        tyro.extras.literal_type_from_choices(all_tasks),
        add_help=False,
        return_unknown_args=True,
        config=mjlab.TYRO_FLAGS,
    )
    cfg = tyro.cli(
        Config,
        args=remaining_args,
        prog=sys.argv[0] + f" {chosen_task}",
        config=mjlab.TYRO_FLAGS,
    )

    checkpoint_path = Path(cfg.checkpoint_file).expanduser().resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    registered_cfg = _REGISTRY[chosen_task].rl_cfg
    if cfg.imitation_checkpoint_file is not None:
        if not hasattr(registered_cfg, "imitation_checkpoint_file"):
            raise ValueError(
                f"Task {chosen_task} does not accept an imitation checkpoint."
            )
        registered_cfg.imitation_checkpoint_file = cfg.imitation_checkpoint_file
    if cfg.rvq_num_active_quantizers is not None:
        if cfg.rvq_num_active_quantizers <= 0:
            raise ValueError("rvq_num_active_quantizers must be positive.")
        if not hasattr(registered_cfg, "rvq_num_active_quantizers"):
            raise ValueError(f"Task {chosen_task} does not configure RVQ quantizers.")
        registered_cfg.rvq_num_active_quantizers = cfg.rvq_num_active_quantizers

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")
    env_cfg = load_env_cfg(chosen_task, play=False)
    agent_cfg = load_rl_cfg(chosen_task)
    episode_length_s = float(
        cfg.episode_length_s
        if cfg.episode_length_s is not None
        else env_cfg.episode_length_s
    )
    if episode_length_s <= 0.0:
        raise ValueError("episode_length_s must be positive.")
    if cfg.num_envs <= 0:
        raise ValueError("num_envs must be positive.")

    command_name, velocity_stages = _configure_evaluation_env(
        env_cfg,
        num_envs=cfg.num_envs,
        episode_length_s=episode_length_s,
        seed=cfg.seed,
    )

    output_dir = (
        Path(cfg.output_dir).expanduser().resolve()
        if cfg.output_dir is not None
        else (
            Path("outputs")
            / "random_command_eval"
            / f"{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{checkpoint_path.stem}"
        ).resolve()
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[INFO] Task: {chosen_task}")
    print(f"[INFO] Checkpoint: {checkpoint_path}")
    print(f"[INFO] Device: {device}; envs: {cfg.num_envs}")
    print(f"[INFO] Output: {output_dir}")

    env: RslRlVecEnvWrapper | None = None
    wall_start = time.perf_counter()
    try:
        raw_env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
        env = RslRlVecEnvWrapper(raw_env, clip_actions=agent_cfg.clip_actions)

        runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
        runner = runner_cls(env, asdict(agent_cfg), device=device)
        actor = getattr(runner.alg, "actor", None)
        resolved_active_quantizers = getattr(actor, "num_active_codebooks", None)
        if cfg.rvq_num_active_quantizers is not None:
            if resolved_active_quantizers is None:
                raise TypeError(
                    "The configured actor does not expose `num_active_codebooks`; "
                    "cannot verify --rvq-num-active-quantizers."
                )
            if int(resolved_active_quantizers) != cfg.rvq_num_active_quantizers:
                raise ValueError(
                    "Active RVQ quantizer mismatch: requested "
                    f"{cfg.rvq_num_active_quantizers}, actor constructed with "
                    f"{resolved_active_quantizers}."
                )
        if resolved_active_quantizers is not None:
            print(f"[INFO] Active RVQ quantizers: {resolved_active_quantizers}")
        runner.load(
            str(checkpoint_path),
            load_cfg={"actor": True},
            strict=True,
            map_location=device,
        )
        policy = runner.get_inference_policy(device=device)

        unwrapped = env.unwrapped
        robot = unwrapped.scene["robot"]
        twist_term = unwrapped.command_manager.get_term(command_name)
        action_term = unwrapped.action_manager.get_term("joint_pos")
        if not hasattr(action_term, "_processed_actions"):
            raise TypeError("joint_pos action term does not expose processed actions.")

        step_dt = float(unwrapped.step_dt)
        num_steps = int(math.ceil(episode_length_s / step_dt))
        num_envs = cfg.num_envs
        num_joints = int(robot.data.joint_pos.shape[1])
        num_actuators = int(robot.data.actuator_force.shape[1])
        action_dim = int(unwrapped.action_manager.total_action_dim)
        reward_names = list(unwrapped.reward_manager.active_terms)

        arrays: dict[str, np.ndarray] = {
            "time_s": np.arange(1, num_steps + 1, dtype=np.float32) * step_dt,
            "valid": np.zeros((num_steps, num_envs), dtype=bool),
            "fell_this_step": np.zeros((num_steps, num_envs), dtype=bool),
            "command": _float_buffer(num_steps, num_envs, 3),
            "next_command": _float_buffer(num_steps, num_envs, 3),
            "command_time_left_s": _float_buffer(num_steps, num_envs),
            "command_counter": np.empty((num_steps, num_envs), dtype=np.int32),
            "is_standing_command": np.zeros((num_steps, num_envs), dtype=bool),
            "base_position_w": _float_buffer(num_steps, num_envs, 3),
            "base_orientation_wxyz": _float_buffer(num_steps, num_envs, 4),
            "base_linear_velocity_b": _float_buffer(num_steps, num_envs, 3),
            "base_angular_velocity_b": _float_buffer(num_steps, num_envs, 3),
            "velocity_error_b": _float_buffer(num_steps, num_envs, 3),
            "orientation_angle_rad": _float_buffer(num_steps, num_envs),
            "joint_position": _float_buffer(num_steps, num_envs, num_joints),
            "joint_velocity": _float_buffer(num_steps, num_envs, num_joints),
            "joint_acceleration": _float_buffer(num_steps, num_envs, num_joints),
            "actuator_torque": _float_buffer(num_steps, num_envs, num_actuators),
            "policy_action": _float_buffer(num_steps, num_envs, action_dim),
            "action": _float_buffer(num_steps, num_envs, action_dim),
            "processed_action": _float_buffer(num_steps, num_envs, action_dim),
            "joint_position_target": _float_buffer(
                num_steps, num_envs, num_joints
            ),
            "reward": _float_buffer(num_steps, num_envs),
            "reward_terms": _float_buffer(
                num_steps, num_envs, len(reward_names)
            ),
        }

        obs = env.get_observations().to(device)
        alive = torch.ones(num_envs, dtype=torch.bool, device=unwrapped.device)
        first_fall_step = np.full(num_envs, -1, dtype=np.int32)
        fall_limit = math.radians(cfg.fall_angle_deg)

        with torch.inference_mode():
            for step in range(num_steps):
                command = twist_term.command.clone()
                command_time_left = twist_term.time_left.clone()
                command_counter = twist_term.command_counter.clone()
                standing = twist_term.is_standing_env.clone()

                policy_action = policy(obs)
                if not isinstance(policy_action, torch.Tensor):
                    raise TypeError(
                        f"Inference policy returned {type(policy_action)}, expected Tensor."
                    )
                action = policy_action.to(unwrapped.device)
                if env.clip_actions is not None:
                    action = torch.clamp(action, -env.clip_actions, env.clip_actions)
                action = action.clone()
                action[~alive] = 0.0

                # Snapshot the exact affine post-processing used by env.step().
                action_term.process_actions(action)
                processed_action = action_term._processed_actions.clone()

                next_obs, reward, _dones, _extras = env.step(action)
                base_lin_vel_b = robot.data.root_link_lin_vel_b
                base_ang_vel_b = robot.data.root_link_ang_vel_b
                actual_twist = torch.stack(
                    (base_lin_vel_b[:, 0], base_lin_vel_b[:, 1], base_ang_vel_b[:, 2]),
                    dim=1,
                )
                orientation_angle = torch.acos(
                    torch.clamp(-robot.data.projected_gravity_b[:, 2], -1.0, 1.0)
                ).abs()
                fell_this_step = alive & (orientation_angle > fall_limit)

                valid_np = _to_numpy(alive)
                fell_np = _to_numpy(fell_this_step)
                arrays["valid"][step] = valid_np
                arrays["fell_this_step"][step] = fell_np
                arrays["command"][step] = _to_numpy(command)
                arrays["next_command"][step] = _to_numpy(twist_term.command)
                arrays["command_time_left_s"][step] = _to_numpy(command_time_left)
                arrays["command_counter"][step] = _to_numpy(command_counter)
                arrays["is_standing_command"][step] = _to_numpy(standing)
                arrays["base_position_w"][step] = _to_numpy(robot.data.root_link_pos_w)
                arrays["base_orientation_wxyz"][step] = _to_numpy(
                    robot.data.root_link_quat_w
                )
                arrays["base_linear_velocity_b"][step] = _to_numpy(base_lin_vel_b)
                arrays["base_angular_velocity_b"][step] = _to_numpy(base_ang_vel_b)
                arrays["velocity_error_b"][step] = _to_numpy(actual_twist - command)
                arrays["orientation_angle_rad"][step] = _to_numpy(orientation_angle)
                arrays["joint_position"][step] = _to_numpy(robot.data.joint_pos)
                arrays["joint_velocity"][step] = _to_numpy(robot.data.joint_vel)
                arrays["joint_acceleration"][step] = _to_numpy(robot.data.joint_acc)
                arrays["actuator_torque"][step] = _to_numpy(robot.data.actuator_force)
                arrays["policy_action"][step] = _to_numpy(policy_action)
                arrays["action"][step] = _to_numpy(action)
                arrays["processed_action"][step] = _to_numpy(processed_action)
                arrays["joint_position_target"][step] = _to_numpy(
                    robot.data.joint_pos_target
                )
                arrays["reward"][step] = _to_numpy(reward)
                arrays["reward_terms"][step] = _to_numpy(
                    unwrapped.reward_manager._step_reward
                )

                new_falls = np.flatnonzero((first_fall_step < 0) & fell_np)
                first_fall_step[new_falls] = step
                alive &= ~fell_this_step
                obs = next_obs.to(device)

                if (step + 1) % max(num_steps // 10, 1) == 0 or step + 1 == num_steps:
                    print(
                        f"[INFO] Step {step + 1}/{num_steps}; "
                        f"active envs: {int(alive.sum().item())}/{num_envs}"
                    )

        rows = [
            _summarize_env(env_id, arrays, step_dt, first_fall_step)
            for env_id in range(num_envs)
        ]
        aggregate = _aggregate_summary(rows)

        npz_path = output_dir / "rollout.npz"
        save_npz = np.savez_compressed if cfg.compress else np.savez
        save_npz(npz_path, **arrays)

        csv_path = output_dir / "per_env_summary.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)

        command_cfg = twist_term.cfg
        metadata = {
            "run": {
                "task_id": chosen_task,
                "checkpoint_file": str(checkpoint_path),
                "imitation_checkpoint_file": cfg.imitation_checkpoint_file,
                "rvq_num_active_quantizers_requested": cfg.rvq_num_active_quantizers,
                "rvq_num_active_quantizers_resolved": resolved_active_quantizers,
                "device": device,
                "seed": cfg.seed,
                "num_envs": num_envs,
                "episode_length_s": episode_length_s,
                "step_dt_s": step_dt,
                "num_steps": num_steps,
                "wall_time_s": time.perf_counter() - wall_start,
                "observation_corruption": False,
                "domain_randomization": False,
                "active_events": list(env_cfg.events),
                "fall_angle_deg": cfg.fall_angle_deg,
            },
            "command": {
                "name": command_name,
                "ranges": {
                    "lin_vel_x": command_cfg.ranges.lin_vel_x,
                    "lin_vel_y": command_cfg.ranges.lin_vel_y,
                    "ang_vel_z": command_cfg.ranges.ang_vel_z,
                },
                "resampling_time_range_s": command_cfg.resampling_time_range,
                "rel_standing_envs": command_cfg.rel_standing_envs,
                "heading_command": command_cfg.heading_command,
                "applied_training_velocity_stages": velocity_stages,
            },
            "names": {
                "joint": list(robot.joint_names),
                "actuator": list(robot.actuator_names),
                "action_target": list(action_term.target_names),
                "reward_term": reward_names,
                "twist_components": ["vx_body_mps", "vy_body_mps", "wz_body_radps"],
            },
            "semantics": {
                "command": "Command used to choose policy_action for this transition.",
                "next_command": "Command after the transition and possible resampling.",
                "velocity_error_b": "actual body twist minus command.",
                "policy_action": "Direct inference-policy output.",
                "action": "Policy action after runner clipping; sent to the environment.",
                "processed_action": "Action after action-term scale and default-pose offset.",
                "joint_position_target": "Position target after action processing.",
                "actuator_torque": "MuJoCo actuator_force in actuator order.",
                "reward": "Total reward scaled by environment dt.",
                "reward_terms": "Weighted per-term reward rates before dt scaling.",
                "valid": "True through the first fall transition; false afterwards.",
            },
            "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
            "summary": aggregate,
            "files": {
                "rollout_npz": str(npz_path),
                "per_env_summary_csv": str(csv_path),
                "metadata_json": str(output_dir / "metadata.json"),
            },
        }
        metadata_path = output_dir / "metadata.json"
        metadata_path.write_text(
            json.dumps(_jsonable(metadata), indent=2, allow_nan=True) + "\n",
            encoding="utf-8",
        )

        print(f"[INFO] Saved {npz_path}")
        print(f"[INFO] Saved {csv_path}")
        print(f"[INFO] Saved {metadata_path}")
        print(
            "[RESULT] "
            f"fall_rate={aggregate['fall_rate']:.3f}, "
            f"linear_error={aggregate['mean_linear_error_norm_mps_mean']:.4f} m/s, "
            f"yaw_error={aggregate['mean_abs_yaw_error_radps_mean']:.4f} rad/s"
        )
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
