# Motion Assets

Place motion assets here.

By default, motion clips are loaded from the dataset referenced in [`motion_data_cfg.yaml`](/home/famadio/Workspace/RL/mjlab_playground/clamp/src/yahmp/config/g1/motion_data_cfg.yaml):

```text
assets/motions/g1_omomo_amass_clean
```

## NPZ format

YAHMP works natively with NPZ motion files.

Supported NPZ layouts:

1. Minimum layout

   - `fps`
   - `root_pos`
   - `root_quat_w`
   - `joint_pos`

    At load time, body/world kinematics are reconstructed with FK.

2. Extended layout

   - `fps`
   - `joint_pos`
   - `joint_vel`
   - `body_pos_w`
   - `body_quat_w`
   - `body_lin_vel_w`
   - `body_ang_vel_w`
   - `body_names`

   This format stores precomputed body/world kinematics directly.

Quaternion convention: `wxyz`

## Compatibility note

YAHMP is also compatible with the PKL motion format used by the TWIST2 dataset.
Those PKLs typically store:

- `root_pos`
- `root_rot`
- `dof_pos`

and use `xyzw` quaternions for `root_rot` by default.

To convert TWIST 2 dataset PKLs to NPZs, use:

```text
src/yahmp/scripts/data/convert_pkl_dataset_to_npz.py
```
