# YAHMP: Yet Another Humanoid Motion tracking Policy

[![mjlab](https://img.shields.io/badge/mjlab-1.2.0-orange.svg)](https://mujocolab.github.io/mjlab/main/index.html)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**YAHMP** is a humanoid general motion tracking policy for the Unitree G1 robot.

The training pipeline builds upon [`mjlab`](https://github.com/mujocolab/mjlab).

<https://github.com/user-attachments/assets/d53fc62a-8915-48dd-914e-4514efe26a1d>

## Installation

First, install [`uv`](https://docs.astral.sh/uv/):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then install the project dependencies with:

```bash
uv sync
```

Finally, verify that the YAHMP environments are correctly installed and visible with:

```bash
uv run list_envs | rg YAHMP
```

## Motion Assets

Download the retargeted [OMOMO](https://github.com/lijiaman/omomo_releaseamss) and [AMASS](https://amass.is.tue.mpg.de/) motions used for training from [this link](https://drive.google.com/file/d/13GOHVAp5plhQCbUCrOnD_aOkqwy46gHi/view?usp=drive_link), extract the archive, and copy the `g1_omomo_amass_clean` folder into `assets/motions/`.

The default motion configuration [`src/yahmp/config/g1/motion_data_cfg.yaml`](src/yahmp/config/g1/motion_data_cfg.yaml) specifies which motions are loaded during training—modify it if you want to use different reference datasets.

## Quick Checks

Smoke-test the base YAHMP scene with a zero-action agent:

```bash
uv run play Mjlab-YAHMP-Unitree-G1 --agent zero
```

## Training

```bash
uv run train Mjlab-YAHMP-Unitree-G1 --env.scene.num-envs 8192
```

## Run Pre-trained Policy

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion-name>.npz
```

## Utilities

Utility scripts for data conversion, ONNX export, deployment, evaluation, and workflow helpers live under [`src/yahmp/scripts`](src/yahmp/scripts). See the dedicated guide at [`src/yahmp/scripts/README.md`](src/yahmp/scripts/README.md).

## Extras

### YAHMP-Future

`Mjlab-YAHMP-Future-Unitree-G1` is a YAHMP variant that augments the base policy with a future-motion encoder. The actor still receives the current motion reference and proprioceptive observations directly, but it also encodes a short horizon of future motion references, which can help anticipate upcoming motion changes.

```bash
uv run train Mjlab-YAHMP-Future-Unitree-G1 --env.scene.num-envs 8192
```

### Teacher-Student pipeline

In addition to `Mjlab-YAHMP-Unitree-G1`, this project also includes environments that together form an example Teacher–Student training pipeline:

- `Mjlab-YAHMP-Teacher-Unitree-G1`: privileged teacher example  
- `Mjlab-YAHMP-Student-RL+Action-Matching-Unitree-G1`: student trained via RL + Action-Matching distillation  
- `Mjlab-YAHMP-Student-RL+KL-Matching-Unitree-G1`: student trained via RL + KL-Matching distillation

```bash
# 1. Train the teacher
uv run train Mjlab-YAHMP-Teacher-Unitree-G1 --env.scene.num-envs 8192

# 2. Train the student from the teacher W&B run
uv run train Mjlab-YAHMP-Student-RL+Action-Matching-Unitree-G1 \
  --env.scene.num-envs 8192 \
  --agent.teacher-wandb-run-path entity/project/teacher_run_id
```

To resume a student run from W&B, use the student run path with `--agent.resume True --wandb-run-path entity/project/student_run_id`. You only need `--agent.teacher-wandb-run-path ...` again if you want to override the embedded teacher with a different one.

## Development

Install the dev tools with:

```bash
uv sync --group dev
```

Format and auto-fix the repo with:

```bash
make format
```
