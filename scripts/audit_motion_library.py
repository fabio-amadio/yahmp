"""B'.1 — Audit the motion library: classify clips by locomotion content.

For each .npz clip in the library:
  - Compute body-frame root linear velocity (vx, vy) and yaw rate (ωz).
  - Compute clip-level statistics (mean magnitudes, fractions).
  - Classify as one of: stationary / walking / turning / mixed / other.

Aggregate output:
  - Per-class clip count and fraction
  - Empirical (vx, vy, ωz) distribution over ALL walking-tagged frames
  - Top-N walking clips by quality (sustained velocity, low jitter)

This script answers: "is omomo_amass_clean actually a locomotion dataset?"
If the walking-class fraction is small or the velocity distribution is narrow,
the imitation backbone has been trained on motions that do NOT cover the
omnidirectional velocity space PPO is trying to learn.
"""

from __future__ import annotations

from pathlib import Path
from dataclasses import dataclass

import numpy as np

MOTIONS_DIR = Path("/home/valerio/yahmp/assets/motions/g1_omomo_amass_clean")

# Classification thresholds.
V_LO = 0.15           # below this, the robot isn't moving
V_HI = 3.0            # above this is unrealistic (mocap noise/jumps)
W_HI = 0.5            # above this we consider it a turning frame
MIN_LEN_S = 1.0       # ignore clips shorter than 1 s


@dataclass
class ClipStats:
    name: str
    duration_s: float
    n_frames: int
    fps: float
    v_xy_mean: float
    v_xy_max: float
    v_xy_q50: float
    w_z_mean: float
    w_z_max: float
    frac_moving: float
    frac_turning: float
    frac_walking_like: float
    label: str


def _quat_to_yaw(q_wxyz: np.ndarray) -> np.ndarray:
    """Yaw from wxyz quaternion. q_wxyz: (T, 4)."""
    w, x, y, z = q_wxyz[..., 0], q_wxyz[..., 1], q_wxyz[..., 2], q_wxyz[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    return np.arctan2(siny_cosp, cosy_cosp)


def _unwrap_yaw(yaw: np.ndarray) -> np.ndarray:
    """Unwrap to remove ±π discontinuities."""
    return np.unwrap(yaw)


def analyze_clip(npz_path: Path) -> ClipStats | None:
    data = np.load(npz_path, allow_pickle=True)
    fps = float(data["fps"].item() if "fps" in data else 30.0)
    dt = 1.0 / fps
    root_pos = data["root_pos"]            # (T, 3)
    root_quat = data["root_quat_w"]        # (T, 4) wxyz
    T = root_pos.shape[0]
    duration_s = T * dt
    if duration_s < MIN_LEN_S or T < 4:
        return None

    # Yaw and yaw-rate.
    yaw = _unwrap_yaw(_quat_to_yaw(root_quat))            # (T,)
    w_z = np.diff(yaw) / dt                                # (T-1,)

    # Body-frame linear velocity.
    # First compute world-frame linear velocity from finite differences.
    v_world = np.diff(root_pos, axis=0) / dt              # (T-1, 3)
    # Rotate horizontal world velocity into body frame using yaw at frame t.
    yaw_t = yaw[:-1]
    c = np.cos(-yaw_t)
    s = np.sin(-yaw_t)
    v_xy_world = v_world[:, :2]
    v_x_body = c * v_xy_world[:, 0] - s * v_xy_world[:, 1]
    v_y_body = s * v_xy_world[:, 0] + c * v_xy_world[:, 1]
    v_xy_mag = np.sqrt(v_x_body**2 + v_y_body**2)
    w_z_mag = np.abs(w_z)

    moving = (v_xy_mag > V_LO) & (v_xy_mag < V_HI)
    turning = w_z_mag > W_HI
    walking_like = moving & (~turning) & (v_xy_mag < 2.5)

    frac_moving = float(moving.mean())
    frac_turning = float(turning.mean())
    frac_walking_like = float(walking_like.mean())

    # Classification.
    if frac_moving < 0.20:
        label = "stationary"
    elif frac_walking_like > 0.55:
        label = "walking"
    elif frac_turning > 0.40:
        label = "turning"
    elif frac_moving > 0.40:
        label = "mixed_locomotion"
    else:
        label = "other"

    return ClipStats(
        name=npz_path.name,
        duration_s=duration_s,
        n_frames=T,
        fps=fps,
        v_xy_mean=float(v_xy_mag[moving].mean()) if moving.any() else 0.0,
        v_xy_max=float(v_xy_mag.max()),
        v_xy_q50=float(np.median(v_xy_mag)),
        w_z_mean=float(w_z_mag[moving].mean()) if moving.any() else 0.0,
        w_z_max=float(w_z_mag.max()),
        frac_moving=frac_moving,
        frac_turning=frac_turning,
        frac_walking_like=frac_walking_like,
        label=label,
    )


def collect_velocity_samples(clips: list[tuple[Path, ClipStats]],
                             labels: tuple[str, ...]) -> np.ndarray:
    """Concatenate (vx_body, vy_body, ωz) over all frames of selected clips."""
    rows = []
    for path, stats in clips:
        if stats.label not in labels:
            continue
        data = np.load(path, allow_pickle=True)
        fps = float(data["fps"].item() if "fps" in data else 30.0)
        dt = 1.0 / fps
        rp = data["root_pos"]
        rq = data["root_quat_w"]
        yaw = _unwrap_yaw(_quat_to_yaw(rq))
        w_z = np.diff(yaw) / dt
        v_w = np.diff(rp, axis=0) / dt
        yt = yaw[:-1]
        c, s = np.cos(-yt), np.sin(-yt)
        vx = c * v_w[:, 0] - s * v_w[:, 1]
        vy = s * v_w[:, 0] + c * v_w[:, 1]
        rows.append(np.stack([vx, vy, w_z], axis=1))
    return np.concatenate(rows, axis=0) if rows else np.zeros((0, 3))


def main() -> None:
    paths = sorted(MOTIONS_DIR.glob("*.npz"))
    print(f"Scanning {len(paths)} clips from {MOTIONS_DIR.name}/ ...")

    results: list[tuple[Path, ClipStats]] = []
    skipped = 0
    for p in paths:
        try:
            stats = analyze_clip(p)
        except Exception as e:
            print(f"  ! failed {p.name}: {e}")
            continue
        if stats is None:
            skipped += 1
            continue
        results.append((p, stats))
    print(f"Analyzed {len(results)} clips ({skipped} skipped: too short/empty).\n")

    # ─── Class distribution ────────────────────────────────────────────────
    from collections import Counter
    labels = [r[1].label for r in results]
    counts = Counter(labels)
    total = len(results)
    print("=== Clip class distribution ===")
    for label in ("walking", "mixed_locomotion", "turning", "stationary", "other"):
        n = counts.get(label, 0)
        print(f"  {label:18s}: {n:4d}  ({n/total*100:5.1f}%)")
    print(f"  TOTAL              : {total:4d}")

    # ─── Duration coverage ─────────────────────────────────────────────────
    total_dur = sum(s.duration_s for _, s in results)
    walking_dur = sum(s.duration_s for _, s in results
                      if s.label in ("walking", "mixed_locomotion"))
    print(f"\nTotal duration: {total_dur:.1f} s "
          f"({total_dur/60:.1f} min)")
    print(f"Walking + mixed_locomotion duration: {walking_dur:.1f} s "
          f"({walking_dur/total_dur*100:.1f}% of total)")

    # ─── Velocity distribution over walking-tagged frames ──────────────────
    walking_samples = collect_velocity_samples(
        results, labels=("walking", "mixed_locomotion")
    )
    if walking_samples.size > 0:
        vx, vy, wz = walking_samples[:, 0], walking_samples[:, 1], walking_samples[:, 2]
        speed = np.sqrt(vx**2 + vy**2)
        print(f"\n=== Velocity distribution (walking + mixed, "
              f"N={len(walking_samples)} frames) ===")
        for name, arr in (("vx", vx), ("vy", vy), ("wz", wz), ("|v_xy|", speed)):
            print(
                f"  {name:6s}: mean={arr.mean():+.3f} "
                f"std={arr.std():.3f}  "
                f"p05={np.percentile(arr, 5):+.3f}  "
                f"p50={np.percentile(arr, 50):+.3f}  "
                f"p95={np.percentile(arr, 95):+.3f}  "
                f"range=[{arr.min():+.2f}, {arr.max():+.2f}]"
            )
        # 2D coverage check vs the locomotion training range.
        in_range_x = (vx >= -0.7) & (vx <= 1.2)
        in_range_y = (vy >= -0.4) & (vy <= 0.4)
        in_range_w = (wz >= -0.3) & (wz <= 0.3)
        print(
            f"\n  Fraction in training range:"
            f"  vx∈[-0.7,1.2]={in_range_x.mean()*100:.1f}%  "
            f"vy∈[-0.4,0.4]={in_range_y.mean()*100:.1f}%  "
            f"wz∈[-0.3,0.3]={in_range_w.mean()*100:.1f}%  "
            f"ALL={((in_range_x & in_range_y & in_range_w).mean()*100):.1f}%"
        )

    # ─── Top-N walking clips for B'.2 ──────────────────────────────────────
    walking = [r for r in results if r[1].label == "walking"]
    walking.sort(
        key=lambda r: (r[1].frac_walking_like, r[1].duration_s),
        reverse=True,
    )
    print(f"\n=== Top walking clips (for B'.2 replay) ===")
    for path, st in walking[:10]:
        print(
            f"  {path.name:55s}  "
            f"dur={st.duration_s:5.1f}s  "
            f"walk_frac={st.frac_walking_like:.2f}  "
            f"v_xy_mean={st.v_xy_mean:.2f}  "
            f"v_xy_max={st.v_xy_max:.2f}"
        )

    if not walking:
        print("  (NONE — dataset contains no clips classified as walking)")

    # ─── Save for downstream use ───────────────────────────────────────────
    out_path = MOTIONS_DIR.parent / "motion_library_audit.npz"
    np.savez(
        out_path,
        names=np.array([r[1].name for r in results]),
        labels=np.array([r[1].label for r in results]),
        durations=np.array([r[1].duration_s for r in results]),
        v_xy_mean=np.array([r[1].v_xy_mean for r in results]),
        v_xy_max=np.array([r[1].v_xy_max for r in results]),
        frac_walking_like=np.array([r[1].frac_walking_like for r in results]),
        velocity_samples=walking_samples,
    )
    print(f"\nSaved per-clip stats and velocity samples → {out_path}")


if __name__ == "__main__":
    main()
