# YAHMP Scripts

This folder contains standalone scripts for data preparation, ONNX export, deployment, evaluation, and small workflow helpers.

General pattern:

```bash
uv run python -m yahmp.scripts.<group>.<script> --help
```

## Layout

- `data/`: motion data conversion and rendering
- `deploy/`: export trained checkpoints to ONNX and run ONNX policies in MuJoCo
- `evaluation/`: batch evaluation and comparison scripts
- `utils/`: small workflow helpers

## Data

### `convert_pkl_dataset_to_npz.py`

Convert PKL motion datasets to NPZ format:

```bash
uv run python -m yahmp.scripts.data.convert_pkl_dataset_to_npz \
  --input-root assets/motions/g1_motions_pkls \
  --output-root assets/motions/g1_motions_npz
```

### `render_reference_motions.py`

Renders one `.mp4` per reference motion clip. The output folder mirrors the motion source subfolder layout.

```bash
uv run python -m yahmp.scripts.data.render_reference_motions \
  --motion-source src/yahmp/config/g1/motion_data_cfg.yaml \
  --output-dir assets/motions/g1_motion_videos \
  --max-motions 10 \
  --overwrite
```

## Deployment

### `export_checkpoint_to_onnx.py`

Export a trained checkpoint to ONNX:

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --checkpoint-file /path/to/model.pt \
  --output-path assets/models/<your-model-name>.onnx
```

You can also export directly from a W&B run:

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --wandb-run-path entity/project/run_id
```

### `run_yahmp_onnx_mujoco.py`

Run an exported YAHMP ONNX policy in MuJoCo:

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion-name>.npz
```

### `run_twist2_onnx_mujoco.py`

Run the original TWIST2 ONNX for reproducibility experiments:

```bash
uv run python -m yahmp.scripts.deploy.run_twist2_onnx_mujoco \
  --onnx-path assets/models/twist2_1017_25k.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion-name>.npz
```

## Evaluation

### `evaluate_yahmp_onnx_success_parallel.py`

Evaluate a base YAHMP ONNX checkpoint over all the motions:

```bash
uv run python -m yahmp.scripts.evaluation.evaluate_yahmp_onnx_success_parallel \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --output-dir assets/logs/yahmp_eval
```

Resume an interrupted evaluation from an existing CSV:

```bash
uv run python -m yahmp.scripts.evaluation.evaluate_yahmp_onnx_success_parallel \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --output-dir assets/logs/yahmp_eval \
  --resume True
```

### `evaluate_twist2_onnx_success_parallel.py`

Evaluate the original TWIST2 ONNX under the same batch evaluation pipeline:

```bash
uv run python -m yahmp.scripts.evaluation.evaluate_twist2_onnx_success_parallel \
  --onnx-path assets/models/twist2_1017_25k.onnx \
  --output-dir assets/logs/twist2_eval
```

Both parallel evaluators use the same falling-only failure criterion.

### `plot_tracking_metrics_boxplot.py`

Plot paired tracking-metric boxplots from two per-motion CSV files:

```bash
uv run python -m yahmp.scripts.evaluation.plot_tracking_metrics_boxplot \
  --twist2-csv assets/logs/twist2_eval/per_motion_success.csv \
  --yahmp-csv assets/logs/yahmp_eval/per_motion_success.csv
```

## Utils

### `play_checkpoint.py`

Play a checkpoint from a W&B run using a modified version of mjlab `play.py` script adapted to handle YAHMP runs that only expose `model_latest.pt` rolling checkpoints.

```bash
uv run python -m yahmp.scripts.utils.play_checkpoint \
  Mjlab-YAHMP-Unitree-G1 \
  --wandb-run-path entity/project/run_id
```
