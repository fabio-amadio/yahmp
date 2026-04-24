# YAHMP Motion

Internal implementation of YAHMP motion commands:

- `JointRefAnchorRpMotionCommandCfg`
- `FutureJointRefAnchorRpMotionCommandCfg`
- `TeacherStudentJointRefAnchorRpMotionCommandCfg`
- `HandBaseMotionCommandCfg`

`HandBase` is kept only as a minimal example command. The YAHMP environments use the joint-reference family.

## Files

- `base.py`
  Shared motion state, frame queries, metrics, and debug hooks.

- `joint_ref.py`
  The base YAHMP command: current joint references with anchor terms and roll/pitch.

- `future_joint_ref.py`
  The future-stacked variant used by the teacher.

- `teacher_student.py`
  Dual-view variant used for distillation. `"default"` matches base YAHMP; `"teacher"` exposes the future-stacked teacher view.

- `hand_base.py`
  Example hand/base command kept as a reference implementation.

- `representations.py`
  Tensor serialization helpers for the active command layouts.

- `sampling.py`
  Resampling and per-step update logic.

- `library.py`
  Motion loading and multi-clip library queries.

- `indexing.py`
  Body-name resolution and mapping checks.

- `debug_visualizer.py`
  Ghost visualization of the current reference.

## Runtime Shape

The split is:

- `base.py`: shared motion state
- `representations.py`: tensor layouts for each command family
- `sampling.py`: how motion state is sampled and advanced

Concrete command classes mostly choose a representation on top of the shared state.
