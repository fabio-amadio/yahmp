"""Play the navigation checkpoint along scripted geometric paths.

Instead of letting ``NavigationGoalCommand`` resample a random goal, this script
overrides the command term's ``_resample_command`` so that the goal walks
through a fixed sequence of world-space waypoints that trace four shapes, in
order:

    1. figure-8   (two lobes, crossing at the origin)
    2. slalom     (out-and-back serpentine)
    3. circle     (circonferenza around the origin)
    4. rosetta    (star: out to a petal tip, pirouette back to the origin, repeat)

After each shape a single goal is sampled AT THE ORIGIN; once the robot reaches
it the next shape begins. The whole sequence loops forever.

Design intent (matches the observed policy behaviour): every hop is short and
the heading keeps changing, so the policy stays in its strong regime (snappy
turns + pirouettes) and never enters a long, linear running stretch where the
gait looks stiff.

The native resample-on-reach is what advances the waypoint pointer; the
time-based resample (``resampling_time_range``) is disabled so it cannot hijack
the path mid-shape.

Usage (num-envs 1 is assumed -- it is a single-robot demo):
  uv run python scripts/play_navigation_scripted.py Mjlab-YAHMP-Navigation-Unitree-G1 \\
      --checkpoint-file logs/rsl_rl/g1_yahmp_navigation/<run>/model_XXXX.pt \\
      --imitation-checkpoint-file /path/to/imitation.pt \\
      --num-envs 1
"""

from __future__ import annotations

import math
import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import mjlab
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
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer

NAV_COMMAND_NAME = "navigate"

Waypoint = tuple[float, float]


# --- Shape generators (all centred on the origin, returning [(x, y), ...]) -----


def _circle(radius: float, n: int) -> list[Waypoint]:
    return [
        (
            radius * math.cos(2.0 * math.pi * i / n),
            radius * math.sin(2.0 * math.pi * i / n),
        )
        for i in range(n)
    ]


# def _figure8(scale: float, n: int) -> list[Waypoint]:
#     # Lemniscate of Gerono: x = a sin t, y = a sin t cos t. Crosses the origin
#     # at t = 0, pi, 2pi; lobes extend along +/-x.
#     # Start at a lobe tip (t=pi/2 -> (scale, 0)) so the first hop from the
#     # origin is a full lobe-radius away (never an instant reach at spawn). The
#     # origin crossings then fall *between* sampled points, not on them.
#     pts: list[Waypoint] = []
#     for i in range(n):
#         t = 0.5 * math.pi + 2.0 * math.pi * i / n
#         pts.append((scale * math.sin(t), scale * math.sin(t) * math.cos(t)))
#     return pts


def _rosetta(radius: float, petals: int) -> list[Waypoint]:
    # Star: origin -> tip_k -> origin -> tip_{k+1} ... Every return through the
    # centre is a ~180deg pirouette (the policy's strong suit).
    pts: list[Waypoint] = []
    for k in range(petals):
        ang = 2.0 * math.pi * k / petals
        pts.append((radius * math.cos(ang), radius * math.sin(ang)))
        pts.append((0.0, 0.0))
    return pts


def _slalom(dx: float, amp: float, gates: int) -> list[Waypoint]:
    # Single-pass serpentine centred on the origin: gates run -x..+x weaving
    # +/-amp. Compact (stays within ~one gate-span of the origin) and every
    # segment is a turning diagonal, never a long straight forward run.
    half = (gates - 1) / 2.0
    return [((i - half) * dx, amp if i % 2 else -amp) for i in range(gates)]


def _dedupe(pts: list[Waypoint], min_spacing: float) -> list[Waypoint]:
    """Drop waypoints closer than ``min_spacing`` to the previous kept one.

    Guarantees consecutive goals sit comfortably beyond ``reach_tol`` so the
    robot never instantly "reaches" the next goal and skips it.
    """
    out: list[Waypoint] = []
    for p in pts:
        if not out or math.hypot(p[0] - out[-1][0], p[1] - out[-1][1]) >= min_spacing:
            out.append(p)
    return out


@dataclass(frozen=True)
class Config:
    checkpoint_file: str
    """Path to the trained navigation .pt checkpoint."""
    imitation_checkpoint_file: str | None = None
    """Imitation .pt (RVQ low-level) checkpoint; required for the YAHMP runner."""
    rvq_num_active_quantizers: int | None = None
    """Active RVQ codebooks; MUST match the trained checkpoint (None = all 8)."""
    num_envs: int = 1
    device: str | None = None
    viewer: Literal["auto", "native", "viser"] = "auto"
    no_terminations: bool = True
    """Disable terminations for a clean, uninterrupted demo loop."""
    motion_file: str | None = None
    """Unused here; kept for parity with the other play scripts."""

    # Shape sizing (metres). Small radii keep every hop short and turny.
    circle_radius: float = 3.0
    circle_radius_stricter: float = 2.0
    # figure8_scale: float = 3.0
    rosetta_radius: float = 1.5
    rosetta_petals: int = 4
    slalom_dx: float = 1.0
    slalom_amp: float = 0.6
    slalom_gates: int = 7
    min_spacing: float = 0.5
    """Minimum spacing between consecutive goals (> reach_tol)."""

    # --- Video recording (native mp4 via mjlab's offscreen renderer) ---------
    video: bool = False
    """Record an mp4 alongside the on-screen viewer. Path markers are included
    (the offscreen renderer runs the same debug_vis as the interactive one)."""
    video_length: int = 1500
    """Frames (= env steps) to record. ~30 s at 50 Hz control."""
    video_start_step: int = 0
    """Steps to wait before recording begins, to skip the spawn transient."""
    video_height: int | None = None
    video_width: int | None = None
    # Camera framing for the recorded mp4 (also sets the on-screen start pose).
    cam_elevation: float | None = None
    """Negative = above looking down. Try -40 for a top-down-ish nav view."""
    cam_distance: float | None = None
    """Metres from the tracked base. Try 4-6 to frame a whole path."""
    cam_azimuth: float | None = None
    """Viewing angle (degrees) around the robot."""


def _build_shapes(cfg: Config) -> list[tuple[str, list[Waypoint]]]:
    """Ordered ``[(name, [(x, y), ...]), ...]`` shapes (origin-anchored)."""
    raw: list[tuple[str, list[Waypoint]]] = [
        # ("figure-8", _figure8(cfg.figure8_scale, n=18)),
        ("slalom", _slalom(cfg.slalom_dx, cfg.slalom_amp, cfg.slalom_gates)),
        ("circle", _circle(cfg.circle_radius, n=12)),
        ("circle", _circle(cfg.circle_radius_stricter, n=8)),
        ("rosetta", _rosetta(cfg.rosetta_radius, cfg.rosetta_petals)),
    ]
    return [(name, _dedupe(pts, cfg.min_spacing)) for name, pts in raw]


# Debug-vis colours (RGBA).
_COLOR_TARGET = (0.95, 0.75, 0.10, 0.95)  # current goal (amber)
_COLOR_PENDING = (0.10, 0.55, 0.95, 0.85)  # not-yet-reached shape points (blue)
_COLOR_ORIGIN = (0.45, 0.45, 0.45, 0.6)  # "home" marker (grey)
_COLOR_ARROW = (0.10, 0.40, 0.90, 0.6)


def _find_nav_term_name(active_terms: list[str]) -> str:
    if NAV_COMMAND_NAME in active_terms:
        return NAV_COMMAND_NAME
    for name in active_terms:
        if "nav" in name.lower() or "goal" in name.lower():
            return name
    raise RuntimeError(
        f"Could not locate the navigation goal command among {active_terms!r}."
    )


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
    print(f"[INFO] Navigation checkpoint: {checkpoint_path}")

    if cfg.imitation_checkpoint_file is not None:
        registered_cfg = _REGISTRY[chosen_task].rl_cfg
        if not hasattr(registered_cfg, "imitation_checkpoint_file"):
            raise ValueError(
                f"Task {chosen_task} runner cfg does not accept "
                "`imitation_checkpoint_file`."
            )
        registered_cfg.imitation_checkpoint_file = cfg.imitation_checkpoint_file
        print(f"[INFO] Imitation checkpoint: {cfg.imitation_checkpoint_file}")

    if cfg.rvq_num_active_quantizers is not None:
        registered_cfg = _REGISTRY[chosen_task].rl_cfg
        if not hasattr(registered_cfg, "rvq_num_active_quantizers"):
            raise ValueError(
                f"Task {chosen_task} runner cfg does not accept "
                "`rvq_num_active_quantizers`."
            )
        registered_cfg.rvq_num_active_quantizers = cfg.rvq_num_active_quantizers
        print(f"[INFO] Active RVQ codebooks: {cfg.rvq_num_active_quantizers}")

    device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    env_cfg = load_env_cfg(chosen_task, play=True)
    agent_cfg = load_rl_cfg(chosen_task)

    # Never auto-reset mid-demo: stretch the episode and (optionally) drop terms.
    env_cfg.episode_length_s = 100000.0
    if cfg.no_terminations:
        env_cfg.terminations = {}
        print("[INFO] Terminations disabled (no_terminations=True)")

    env_cfg.scene.num_envs = cfg.num_envs

    # Camera framing: applies to both the on-screen viewer and the recorded mp4.
    if cfg.cam_elevation is not None:
        env_cfg.viewer.elevation = cfg.cam_elevation
    if cfg.cam_distance is not None:
        env_cfg.viewer.distance = cfg.cam_distance
    if cfg.cam_azimuth is not None:
        env_cfg.viewer.azimuth = cfg.cam_azimuth
    if cfg.video_height is not None:
        env_cfg.viewer.height = cfg.video_height
    if cfg.video_width is not None:
        env_cfg.viewer.width = cfg.video_width

    render_mode = "rgb_array" if cfg.video else None
    env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

    if cfg.video:
        from mjlab.utils.wrappers import VideoRecorder

        video_folder = checkpoint_path.parent / "videos" / "scripted_nav"
        env = VideoRecorder(
            env,
            video_folder=video_folder,
            step_trigger=lambda s: s == cfg.video_start_step,
            video_length=cfg.video_length,
            disable_logger=True,
        )
        print(
            f"[INFO] Recording mp4 -> {video_folder} "
            f"(starts at step {cfg.video_start_step}, {cfg.video_length} frames). "
            "The offscreen camera follows env_cfg.viewer (cam-* flags), not the mouse."
        )

    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner_cls = load_runner_cls(chosen_task) or MjlabOnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(
        str(checkpoint_path),
        load_cfg={"actor": True},
        strict=True,
        map_location=device,
    )
    policy = runner.get_inference_policy(device=device)

    cmd_mgr = env.unwrapped.command_manager
    nav_name = _find_nav_term_name(cmd_mgr.active_terms)
    print(f"[INFO] Driving navigation command term: {nav_name!r}")
    nav_term = cmd_mgr.get_term(nav_name)

    # World-frame anchor for the paths = env-0 origin (the "origin" the robot
    # returns to between shapes). Falls back to (0, 0).
    scene = env.unwrapped.scene
    env_origins = getattr(scene, "env_origins", None)
    if env_origins is not None:
        origin_xy = env_origins[0, :2].to(device)
    else:
        origin_xy = torch.zeros(2, device=device)
    print(
        f"[INFO] Path origin (world): ({float(origin_xy[0]):.2f}, {float(origin_xy[1]):.2f})"
    )

    shapes = _build_shapes(cfg)
    print(
        "[INFO] Shapes (looping): "
        + ", ".join(f"{n}({len(p)})" for n, p in shapes)
        + " -- each closed by an origin return."
    )

    # Kill the time-based resample so only on-reach advancement drives the path.
    if hasattr(nav_term.cfg, "resampling_time_range"):
        nav_term.cfg.resampling_time_range = (1.0e9, 1.0e9)

    # State machine driving one shape at a time.
    #   phase "shape":  heading to ``pending[0]``; reached points are popped.
    #   phase "origin": heading to the origin after a shape is fully consumed.
    state: dict = {"shape_idx": 0, "pending": [], "phase": "shape", "init": False}

    def _load_shape(idx: int) -> None:
        name, pts = shapes[idx]
        state["shape_idx"] = idx
        state["pending"] = [tuple(p) for p in pts]  # full shape -> all visible
        state["phase"] = "shape"
        print(f"[shape] {name}: spawned {len(pts)} points")

    def _current_target() -> Waypoint:
        if state["phase"] == "shape" and state["pending"]:
            return state["pending"][0]
        return (0.0, 0.0)  # origin

    def _apply_target(env_ids: torch.Tensor) -> None:
        tx, ty = _current_target()
        nav_term.target_pos_w[env_ids, 0] = origin_xy[0] + tx
        nav_term.target_pos_w[env_ids, 1] = origin_xy[1] + ty
        root_xy = nav_term.robot.data.root_link_pos_w[env_ids, :2]
        new_dist = torch.norm(root_xy - nav_term.target_pos_w[env_ids], dim=-1)
        nav_term.distance[env_ids] = new_dist
        nav_term.prev_distance[env_ids] = new_dist
        nav_term.progress[env_ids] = 0.0
        nav_term.time_left[env_ids] = 1.0e9  # belt-and-suspenders vs the timer
        nav_term._refresh_command()

    def scripted_resample_command(env_ids: torch.Tensor) -> None:
        # Called by CommandTerm._resample when the current goal is reached (and
        # once at init). Advance the state machine, then re-point the goal.
        if not state["init"]:
            _load_shape(0)
            state["init"] = True
        elif state["phase"] == "shape":
            if state["pending"]:
                state["pending"].pop(0)  # reached point -> disappears
            if not state["pending"]:
                state["phase"] = "origin"  # whole shape consumed -> go home
        else:  # phase == "origin": reached the origin -> next shape
            _load_shape((state["shape_idx"] + 1) % len(shapes))
        _apply_target(env_ids)

    nav_term._resample_command = scripted_resample_command  # type: ignore[assignment]

    def scripted_debug_vis(visualizer) -> None:
        # Redrawn every frame (the viewer zeroes ngeom first): the current shape's
        # not-yet-reached points are shown, the active goal highlighted, and the
        # origin marked. Popped points simply stop being drawn -> they vanish.
        env_indices = list(visualizer.get_env_indices(nav_term.num_envs))
        if not env_indices:
            return
        ox, oy = float(origin_xy[0]), float(origin_xy[1])
        pending = list(state["pending"])  # snapshot vs the step thread
        phase = state["phase"]
        root = nav_term.robot.data.root_link_pos_w.cpu().numpy()
        for b in env_indices:
            for j, (x, y) in enumerate(pending):
                is_target = phase == "shape" and j == 0
                if is_target:
                    visualizer.add_sphere(
                        center=(ox + x, oy + y, 0.06),
                        radius=0.16,
                        color=_COLOR_TARGET,
                        label=f"nav_pt_{b}_{j}",
                    )
            if phase == "origin":
                visualizer.add_sphere(
                    center=(ox, oy, 0.06),
                    radius=0.16,
                    color=_COLOR_TARGET,
                    label=f"nav_origin_{b}",
                )
            else:
                visualizer.add_sphere(
                    center=(ox, oy, 0.02),
                    radius=0.07,
                    color=_COLOR_ORIGIN,
                    label=f"nav_home_{b}",
                )
            tx, ty = _current_target()
            visualizer.add_arrow(
                (float(root[b, 0]), float(root[b, 1]), 0.06),
                (ox + tx, oy + ty, 0.06),
                color=_COLOR_ARROW,
                width=0.012,
                label=f"nav_vec_{b}",
            )

    nav_term._debug_vis_impl = scripted_debug_vis  # type: ignore[assignment]
    # Force the first scripted goal now (the construction-time reset already ran
    # with random sampling before the patch was installed).
    all_ids = torch.arange(cfg.num_envs, device=device)
    scripted_resample_command(all_ids)

    if cfg.viewer == "auto":
        has_display = bool(
            os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")
        )
        resolved_viewer = "native" if has_display else "viser"
    else:
        resolved_viewer = cfg.viewer

    print(f"[INFO] Viewer: {resolved_viewer}")
    if resolved_viewer == "native":
        NativeMujocoViewer(env, policy).run()
    elif resolved_viewer == "viser":
        ViserPlayViewer(env, policy).run()
    else:
        raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

    env.close()


if __name__ == "__main__":
    main()
