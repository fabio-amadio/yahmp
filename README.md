# YAHMP: Yet Another Humanoid Motion Tracking Policy

[![mjlab](https://img.shields.io/badge/mjlab-1.2.0-orange.svg)](https://mujocolab.github.io/mjlab/main/index.html)
[![uv](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)

YAHMP is an open-source framework for training, evaluating, and deploying general motion tracking policies on the Unitree G1. It is built on top of [mjlab](https://github.com/mujocolab/mjlab) and includes the nominal policy, the design variants studied in the accompanying paper, and a privileged teacher-student training pipeline.

<https://github.com/user-attachments/assets/d53fc62a-8915-48dd-914e-4514efe26a1d>

The supplementary video is available at <https://youtu.be/BH6FpQzwm8M>.

## Installation

YAHMP supports Python 3.11--3.13. Install [`uv`](https://docs.astral.sh/uv/) and create the project environment:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/fabio-amadio/yahmp.git
cd yahmp
uv sync
```

Verify that the tasks are registered:

```bash
uv run list_envs | grep YAHMP
```

## Motion Assets

Download the retargeted [OMOMO](https://github.com/lijiaman/omomo_release) and [AMASS](https://amass.is.tue.mpg.de/) motions from [Google Drive](https://drive.google.com/file/d/13GOHVAp5plhQCbUCrOnD_aOkqwy46gHi/view?usp=sharing). Extract the archive so that the dataset is located at:

```text
assets/motions/g1_omomo_amass_clean/
```

The default motion configuration is [`src/yahmp/config/g1/motion_data_cfg.yaml`](src/yahmp/config/g1/motion_data_cfg.yaml). The supported motion formats are described in [`assets/motions/README.md`](assets/motions/README.md).

## Tasks

The nominal policy uses a 10-step observation history, joint-position and joint-velocity references, residual actions, and mechanics-based PD gains and action scales.

| Configuration | Task ID | Difference from nominal |
| --- | --- | --- |
| Nominal | `Mjlab-YAHMP-Unitree-G1` | - |
| No history | `Mjlab-YAHMP-NoHistory-Unitree-G1` | Removes observation history and its encoder |
| History-20 | `Mjlab-YAHMP-History20-Unitree-G1` | Uses a 20-step observation history |
| Position-reference-only | `Mjlab-YAHMP-QOnly-Unitree-G1` | Removes reference joint velocities |
| No residual | `Mjlab-YAHMP-NoResidual-Unitree-G1` | Applies actions around a fixed default posture |
| Stiffer fixed-scale | `Mjlab-YAHMP-StiffPD-Unitree-G1` | Uses stiffer PD gains and constant action scales |
| Hand-force randomization | `Mjlab-YAHMP-HandForceRand-Unitree-G1` | Applies randomized feasible forces at the hands during training |
| Privileged teacher | `Mjlab-YAHMP-Teacher-Unitree-G1` | Trains the privileged teacher used by the student |
| Teacher-student | `Mjlab-YAHMP-Student-RL+KL-Matching-Unitree-G1` | Trains the deployable student with PPO and KL matching |

## Training

Run a short local smoke test without W&B:

```bash
uv run train Mjlab-YAHMP-Unitree-G1 \
  --env.scene.num-envs 64 \
  --agent.max-iterations 2 \
  --agent.logger tensorboard
```

Train the nominal policy with the paper configuration:

```bash
uv run train Mjlab-YAHMP-Unitree-G1 \
  --env.scene.num-envs 8192
```

Training uses W&B by default and runs for 20,000 PPO iterations. Replace the task ID with any entry in the table to train that configuration. To keep a full run local, add `--agent.logger tensorboard`.

### Teacher-Student Training

First train the privileged teacher:

```bash
uv run train Mjlab-YAHMP-Teacher-Unitree-G1 \
  --env.scene.num-envs 8192
```

Then train the student from the teacher checkpoint:

```bash
uv run train Mjlab-YAHMP-Student-RL+KL-Matching-Unitree-G1 \
  --env.scene.num-envs 8192 \
  --agent.teacher-checkpoint-file /path/to/teacher/model_latest.pt
```

A teacher stored on W&B can instead be selected with `--agent.teacher-wandb-run-path entity/project/run_id`.

## Playback

Check the environment with a zero-action policy:

```bash
uv run play Mjlab-YAHMP-Unitree-G1 --agent zero
```

Play a local training checkpoint:

```bash
uv run play Mjlab-YAHMP-Unitree-G1 \
  --checkpoint-file /path/to/model_latest.pt
```

Use `--motion-file /path/to/motion.npz` to select a motion and `--video True` to record the rollout. A checkpoint stored on W&B can be loaded with `--wandb-run-path entity/project/run_id`.

## ONNX Export

Export a trained model to ONNX format:

```bash
uv run python -m yahmp.scripts.deploy.export_checkpoint_to_onnx \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --checkpoint-file /path/to/model_latest.pt \
  --output-path assets/models/exported_policy.onnx
```

## Evaluation and Utilities

The repository includes the nominal policy at [`assets/models/g1_yahmp.onnx`](assets/models/g1_yahmp.onnx).

Run the nominal policy directly in MuJoCo:

```bash
uv run python -m yahmp.scripts.deploy.run_yahmp_onnx_mujoco \
  --task-id Mjlab-YAHMP-Unitree-G1 \
  --onnx-path assets/models/g1_yahmp.onnx \
  --motion-file assets/motions/g1_omomo_amass_clean/<motion-name>.npz
```

Evaluate the nominal policy over the configured motion dataset:

```bash
uv run python -m yahmp.scripts.evaluation.evaluate_policy \
  --onnx-path assets/models/g1_yahmp.onnx \
  --output-dir assets/logs/yahmp_eval
```

Motion rendering, checkpoint export, and evaluation commands are documented in [`src/yahmp/scripts/README.md`](src/yahmp/scripts/README.md).

## Citation

This repository accompanies the paper *What Matters in Humanoid General Motion Tracking? An Empirical Study*, accepted at Humanoids 2026:

```bibtex
@inproceedings{amadio2026what,
  title = {What Matters in Humanoid General Motion Tracking? An Empirical Study},
  author = {Amadio, Fabio and Mingo Hoffman, Enrico},
  booktitle = {IEEE-RAS International Conference on Humanoid Robots (Humanoids)},
  year = {2026}
}
```

## Development

Install the development dependencies and run the checks with:

```bash
uv sync --group dev
make format
make check
```

## License

YAHMP is released under the [Apache 2.0 license](LICENSE).
