"""Convert ember-lab-berkeley/LAFAN-G1 npz files to the YAHMP motion format.

LAFAN-G1 schema (per file):
  - fps                       : (1,) float32
  - dof_names                 : (29,) str       (G1 joint names)
  - body_names                : (30,) str       (first one is "pelvis")
  - dof_positions             : (T, 29) float32
  - dof_velocities            : (T, 29) float32
  - body_positions            : (T, 30, 3) float32   world frame
  - body_rotations            : (T, 30, 4) float32   wxyz, world frame
  - body_linear_velocities    : (T, 30, 3) float32
  - body_angular_velocities   : (T, 30, 3) float32

YAHMP minimal schema produced here (the loader will FK body_* via mjlab URDF):
  - fps                       : scalar
  - joint_pos    = dof_positions
  - joint_vel    = dof_velocities                       (ground truth, free of charge)
  - root_pos     = body_positions[:, 0, :]              (pelvis = body[0])
  - root_quat_w  = body_rotations[:, 0, :]              (wxyz already)

Per-file defensive checks (any failure → file is rejected, NOT silently corrupted):
  - body_names[0] == "pelvis"
  - tuple(dof_names) matches the expected G1 joint order (bit-identical, no permutation)
  - fps > 0
  - num_frames >= 30 (>= 1 s at 30 fps)
  - no NaN/Inf anywhere
  - quaternion norms within [0.95, 1.05] (loader will renormalize, just a sanity floor)

Atomic write via temp file + rename, so a Ctrl-C halfway can never leave a
half-written file in the output directory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

EXPECTED_DOF_NAMES = (
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
NUM_DOF = len(EXPECTED_DOF_NAMES)
MIN_FRAMES = 30


def _decode_names(arr: np.ndarray) -> tuple[str, ...]:
    out: list[str] = []
    for v in arr.flatten().tolist():
        out.append(v.decode("utf-8") if isinstance(v, bytes) else str(v))
    return tuple(out)


def convert_one(in_path: Path, out_path: Path) -> tuple[bool, str]:
    """Convert a single LAFAN-G1 npz into YAHMP minimal format.

    Returns (ok, info_or_error_msg).
    """
    try:
        with np.load(in_path, allow_pickle=False) as data:
            required = {
                "fps",
                "dof_names",
                "body_names",
                "dof_positions",
                "dof_velocities",
                "body_positions",
                "body_rotations",
            }
            missing = required - set(data.files)
            if missing:
                return False, f"missing keys: {sorted(missing)}"

            fps = float(np.asarray(data["fps"]).reshape(-1)[0])
            if not np.isfinite(fps) or fps <= 0:
                return False, f"invalid fps={fps}"

            dof_names = _decode_names(data["dof_names"])
            if tuple(dof_names) != EXPECTED_DOF_NAMES:
                first_mismatch = next(
                    (
                        i
                        for i, (a, b) in enumerate(zip(dof_names, EXPECTED_DOF_NAMES))
                        if a != b
                    ),
                    -1,
                )
                return (
                    False,
                    f"dof_names order mismatch (first diff at idx {first_mismatch}: "
                    f"got '{dof_names[first_mismatch] if first_mismatch>=0 else '?'}' "
                    f"expected '{EXPECTED_DOF_NAMES[first_mismatch] if first_mismatch>=0 else '?'}')",
                )

            body_names = _decode_names(data["body_names"])
            if not body_names or body_names[0] != "pelvis":
                return False, f"body_names[0]={body_names[0] if body_names else '<empty>'} (expected 'pelvis')"

            dof_positions = np.asarray(data["dof_positions"], dtype=np.float32)
            dof_velocities = np.asarray(data["dof_velocities"], dtype=np.float32)
            body_positions = np.asarray(data["body_positions"], dtype=np.float32)
            body_rotations = np.asarray(data["body_rotations"], dtype=np.float32)

            T = dof_positions.shape[0]
            if T < MIN_FRAMES:
                return False, f"too short: {T} frames (< {MIN_FRAMES})"
            if dof_positions.shape != (T, NUM_DOF):
                return False, f"dof_positions shape {dof_positions.shape} != (T, {NUM_DOF})"
            if dof_velocities.shape != (T, NUM_DOF):
                return False, f"dof_velocities shape {dof_velocities.shape} != (T, {NUM_DOF})"
            if body_positions.shape[0] != T or body_positions.shape[2] != 3:
                return False, f"body_positions shape {body_positions.shape} unexpected"
            if body_rotations.shape[0] != T or body_rotations.shape[2] != 4:
                return False, f"body_rotations shape {body_rotations.shape} unexpected"

            # Extract pelvis (= body 0) for the YAHMP root_*
            root_pos = body_positions[:, 0, :].astype(np.float64, copy=True)
            root_quat_w = body_rotations[:, 0, :].astype(np.float64, copy=True)

            # Finiteness checks on everything we will write out
            for name, arr in (
                ("dof_positions", dof_positions),
                ("dof_velocities", dof_velocities),
                ("root_pos", root_pos),
                ("root_quat_w", root_quat_w),
            ):
                if not np.isfinite(arr).all():
                    bad = int((~np.isfinite(arr)).sum())
                    return False, f"{name} contains {bad} non-finite values"

            # Quaternion sanity (loader will renormalize, but reject pathologies)
            q_norms = np.linalg.norm(root_quat_w, axis=-1)
            min_n, max_n = float(q_norms.min()), float(q_norms.max())
            if min_n < 0.95 or max_n > 1.05:
                return False, f"root_quat_w norms out of range: [{min_n:.3f}, {max_n:.3f}]"
    except Exception as exc:
        return False, f"load/parse error: {exc!r}"

    # Atomic write: tmp file in same dir, then rename. We pass an open file
    # object to np.savez to suppress its auto-append of ".npz" to the path.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(out_path.name + ".tmp")
    try:
        with open(tmp_path, "wb") as fh:
            np.savez(
                fh,
                fps=np.array([fps], dtype=np.float32),
                joint_pos=dof_positions,
                joint_vel=dof_velocities,
                root_pos=root_pos,
                root_quat_w=root_quat_w,
            )
        tmp_path.replace(out_path)
    except Exception as exc:
        if tmp_path.exists():
            try:
                tmp_path.unlink()
            except OSError:
                pass
        return False, f"write error: {exc!r}"

    duration_s = T / fps
    return True, f"T={T} ({duration_s:.1f}s @ {fps:.0f}fps)"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--input-dir", required=True, type=Path)
    ap.add_argument("--output-dir", required=True, type=Path)
    ap.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing output files.",
    )
    args = ap.parse_args()

    in_dir = args.input_dir.expanduser().resolve()
    out_dir = args.output_dir.expanduser().resolve()

    if not in_dir.exists() or not in_dir.is_dir():
        print(f"ERROR: input dir does not exist: {in_dir}", file=sys.stderr)
        return 2

    files = sorted(in_dir.glob("*.npz"))
    if not files:
        print(f"ERROR: no .npz files found in {in_dir}", file=sys.stderr)
        return 2

    print(f"Converting {len(files)} files: {in_dir} -> {out_dir}")
    n_ok, n_skip, n_fail = 0, 0, 0
    failures: list[tuple[str, str]] = []

    for f in files:
        out_path = out_dir / f.name
        if out_path.exists() and not args.force:
            n_skip += 1
            print(f"  SKIP  {f.name} (output exists; use --force to overwrite)")
            continue
        ok, info = convert_one(f, out_path)
        if ok:
            n_ok += 1
            print(f"  OK    {f.name}  {info}")
        else:
            n_fail += 1
            failures.append((f.name, info))
            print(f"  FAIL  {f.name}  {info}")

    print()
    print(f"Done. converted={n_ok} skipped={n_skip} failed={n_fail}")
    if failures:
        print("Failures:")
        for name, info in failures:
            print(f"  - {name}: {info}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
