# YAHMP Scripts

Run each script as a Python module and use `--help` to inspect all options:

```bash
uv run python -m yahmp.scripts.<group>.<script> --help
```

## Motion Data

### `expand_npz_motion_dataset.py`

Expand compact NPZ motions into deploy-friendly NPZs with precomputed joint velocities and body world kinematics:

```bash
uv run python -m yahmp.scripts.data.expand_npz_motion_dataset \
  --input-root assets/motions/g1_omomo_amass_clean \
  --output-root assets/motions/g1_omomo_amass_clean_full
```

Render reference motions from the configured dataset:

```bash
uv run python -m yahmp.scripts.data.render_reference_motions \
  --output-dir assets/motions/rendered \
  --max-motions 10
```

## Checkpoints and ONNX

Play a local checkpoint:

```bash
uv run play Mjlab-YAHMP-Unitree-G1 \
  --checkpoint-file /path/to/model_latest.pt
```

Download and play a checkpoint from W&B:

```bash
uv run play Mjlab-YAHMP-Unitree-G1 \
  --wandb-run-path entity/project/run_id
```

Export a local checkpoint to ONNX:

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --checkpoint-file /path/to/model_latest.pt \
  --output-path assets/models/exported_policy.onnx
```

The exporter also accepts `--wandb-run-path entity/project/run_id` in place of `--checkpoint-file`.

Run an ONNX policy in MuJoCo:

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/exported_policy.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion-name>.npz
```

## Evaluation

Evaluate the nominal policy on all motions in the configured dataset:

```bash
uv run python -m yahmp.scripts.evaluation.evaluate_policy \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --num-envs 256 \
  --output-dir assets/logs/yahmp_eval
```
