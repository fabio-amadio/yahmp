# Top-Level Scripts

## Random-command locomotion evaluation

`evaluate_random_commands.py` evaluates a locomotion checkpoint headlessly with
parallel environments. It uses the training command sampler at its cumulative
last curriculum stage while disabling observation noise, pushes, and domain
randomization.

```bash
uv run python scripts/evaluate_random_commands.py \
  Mjlab-YAHMP-Locomotion-Unitree-G1 \
  --checkpoint-file /home/famadio/Workspace/2026-06-24_10-05-40_Loco-Quant-5/model_4999.pt \
  --imitation-checkpoint-file assets/models/multi_task/imitation_rvq_model.pt \
  --rvq-num-active-quantizers 5 \
  --num-envs 64 \
  --output-dir outputs/paper_random_commands
```

The default rollout duration is the environment's training episode length.
Override it with, for example, `--episode-length-s 30`.

The output directory contains:

- `rollout.npz`: per-step commands, tracking errors, robot state, actuator
  torques, policy actions, processed actions, joint targets, and rewards.
- `per_env_summary.csv`: per-environment tracking, fall, action, and torque
  statistics.
- `metadata.json`: command configuration, resolved active quantizer count,
  tensor names and shapes, and aggregate metrics.

Dense rollout arrays use `[step, environment, ...]` axis order. Run
`uv run python scripts/evaluate_random_commands.py Mjlab-YAHMP-Locomotion-Unitree-G1 --help`
for all options.

## Quantizer ramp viewer

`play_quantizer_ramp.py` opens a viewer for one locomotion checkpoint and sends
the robot a forward command ramp.
Pass the policy checkpoint explicitly. The active quantizer count can also be
passed explicitly; if omitted, the script tries to infer it from checkpoint
paths containing names like `Loco-Quant-5` or `quant_5`.

```bash
uv run python scripts/play_quantizer_ramp.py \
  Mjlab-YAHMP-Locomotion-Unitree-G1 \
  --checkpoint-file /home/famadio/Workspace/2026-06-24_10-05-40_Loco-Quant-5/model_4999.pt \
  --imitation-checkpoint-file assets/models/multi_task/imitation_rvq_model.pt \
  --rvq-num-active-quantizers 5
```

Useful overrides:

```bash
uv run python scripts/play_quantizer_ramp.py \
  Mjlab-YAHMP-Locomotion-Unitree-G1 \
  --checkpoint-file /home/famadio/Workspace/2026-06-24_10-05-40_Loco-Quant-5/model_4999.pt \
  --imitation-checkpoint-file assets/models/multi_task/imitation_rvq_model.pt \
  --start-vx 0.3 \
  --end-vx 2.5 \
  --ramp-s 25
```

Headless video:

```bash
uv run python scripts/play_quantizer_ramp.py \
  Mjlab-YAHMP-Locomotion-Unitree-G1 \
  --checkpoint-file /home/famadio/Workspace/2026-06-24_10-05-40_Loco-Quant-5/model_4999.pt \
  --imitation-checkpoint-file assets/models/multi_task/imitation_rvq_model.pt \
  --rvq-num-active-quantizers 5 \
  --video True \
  --video-length-s 33 \
  --video-dir outputs/ramp_videos \
  --video-width 1280 \
  --video-height 720
```
