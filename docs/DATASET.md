# Demo dataset format

`python -m tasks.generate_demos --task <task>` writes ManiSkill `RecordEpisode` files to
`<record-dir>/<EnvId>/scripted_pd_ee_target_pose.{h5,json}`. Only successful episodes are saved.
The example shapes below are from a 211-step `StackCubeColoredFR3-v1` episode recorded with `--obs-mode rgb`.

## Timing and synchronization

| Rate | Value |
|---|---|
| Physics | 100 Hz (`sim_freq`) |
| Control, observations, cameras, actions | 20 Hz (`control_freq`), one sample per `env.step` |

Each control step applies one action, runs 5 physics substeps with that PD target, and then reads
joints, TCP pose, object states and both camera images from the same simulator state, except that each
camera image is `latency.camera_seconds` old (the front camera one control step), as measured on the real
cell. There is no exposure time or jitter. The actuation delays of
the real FR3 are modelled instead: the TCP follows the commanded pose with a 0.39 s lag, and the gripper
acts on a command 0.3 s late (`robot` in `config/config.yaml`).

For an episode with `T` actions:

- `obs/*` and `env_states/*` have `T + 1` entries: index 0 is the state after reset, index `t` is the state the policy saw before `actions[t]`.
- `actions`, `rewards`, `success`, `fail`, `terminated`, `truncated` have `T` entries: index `t` is the action at step `t` and its outcome.

## HDF5 contents (`traj_<episode_id>/`)

| Dataset | Shape | Meaning |
|---|---|---|
| `actions` | `(T, 8)` float32 | `pd_ee_target_pose` action, see below |
| `obs/agent/qpos` | `(T+1, 9)` | `fr3_joint1..7` (rad), `fr3_finger_joint1..2` (m, 0 closed to 0.04 open) |
| `obs/agent/qvel` | `(T+1, 9)` | joint velocities (rad/s, m/s) |
| `obs/extra/tcp_pose` | `(T+1, 7)` | TCP (`fr3_hand_tcp`) world pose: position (m) + quaternion **wxyz** |
| `obs/extra/<object>_pose`, `...pos` | `(T+1, 7 or 3)` | object poses and relative vectors; **only when obs mode includes state** (e.g. `rgb+state`) |
| `obs/sensor_data/base_camera/rgb` | `(T+1, H, W, 3)` uint8 | fixed side camera |
| `obs/sensor_data/wrist_camera/rgb` | `(T+1, H, W, 3)` uint8 | camera mounted on `fr3_hand` |
| `obs/sensor_param/<camera>/intrinsic_cv` | `(T+1, 3, 3)` | OpenCV intrinsics |
| `obs/sensor_param/<camera>/extrinsic_cv` | `(T+1, 3, 4)` | OpenCV world-to-camera; changes every step for the wrist camera |
| `obs/sensor_param/<camera>/cam2world_gl` | `(T+1, 4, 4)` | OpenGL camera-to-world |
| `env_states/actors/<name>` | `(T+1, 13)` | position 3, quaternion wxyz 4, linear velocity 3, angular velocity 3; includes distractors when enabled |
| `env_states/articulations/fr3` | `(T+1, 31)` | root pose 7, root velocity 6, qpos 9, qvel 9 |
| `rewards` | `(T,)` | dense reward |
| `success`, `fail` | `(T,)` bool | task success and irrecoverable failure (object off the table) |
| `terminated`, `truncated` | `(T,)` bool | episode end by success/fail, or by the step limit |

Camera resolution is `cameras.width/height` in `config/data.yaml` and is shared by all tasks.
`env_states` exists to restore exact simulator states for replay; the camera marker actors listed there are unused helpers.

## Action (`pd_ee_target_pose`, 8-D)

| Index | Meaning |
|---|---|
| 0–2 | absolute TCP target position in the **robot base frame** (m) |
| 3–6 | absolute TCP target orientation in the robot base frame, quaternion **wxyz** |
| 7 | Gripper target per finger in meters, **absolute**, 0.0 closed to 0.04 open |

The robot base frame is the world frame rotated +90° about z (the FR3 faces world +y). This is the
command the real policies send to `fr3_teleop`, and the TCP chases it the way the real impedance stack does:
it stays still until the command is `ee_deadband_*` away, closes the rest first order (`ee_lag_seconds`,
`ee_rotation_lag_seconds`) up to `ee_max_speed` / `ee_max_angular_speed_degrees`, and when blocked pushes with
at most `ee_error_clip_*` of offset on softer joint drives (`ee_target_stiffness`), about the real 20 N.
`CartesianPlanner` pushes its command past the target, like the operator, to overcome the deadband. The recorded action therefore
runs ahead of `tcp_pose`, as `actions/cartesian_position` runs ahead of the state in the real data.
`pd_ee_delta_pose` is still available; `CartesianPlanner` drives either mode.

## Training export (`config/data.yaml`)

When `export.enabled` is true, `generate_demos` writes `<record-dir>/<EnvId>/export_<format>_<hz>hz/`
after recording `recording.num_episodes` successful episodes (`make demos <task> N=...` overrides the count).
The export keeps every `20 / hz`-th recorded step (`hz` must divide 20) and replaces the per-step delta
action with absolute targets. Each exported step pairs one observation with the action that follows it;
the final observation of each episode has no action and is dropped, so every array has `K` entries.

Both formats contain the same per-step keys:

| Key | Shape per step | Meaning |
|---|---|---|
| `timestamp` | `()` float32 | seconds since reset |
| `qpos` | `(9,)` | 7 arm joints (rad) + 2 fingers (m) |
| `tcp_pose` | `(7,)` | TCP pose in the **robot base frame**, xyz + quaternion wxyz |
| `qvel` | `(9,)` | joint velocities (rad/s, m/s) |
| `action_gripper` | `(1,)` | absolute finger target (m) commanded at the start of the interval |
| `action_ee_pose` | `(7,)` | TCP pose reached at the next exported step (robot base frame) |
| `action_joint_position` | `(7,)` | arm joints at the next exported step |
| `action_ee_target` | `(7,)` | commanded TCP target at the start of the interval (robot base frame, wxyz) |
| `success` | `()` bool | task success after the interval |
| `<camera>` | `(H, W, 3)` uint8 | cameras in `export.cameras` (zarr only; HDF5 stores them as videos) |

`action_*` keys are written only when listed in `export.actions`. Action chunking is left to the training code.

### `format: hdf5`

```
export_hdf5_<hz>hz/
  episodes.h5                 traj_N/<key> datasets, traj_N attrs: prompt, episode_seed, <camera>_video
  videos/<camera>/traj_N.mp4  one H.264 video per camera per episode, <hz> fps, frame k = step k
```

Videos are lossy (about 40 dB PSNR against the rendered frames). Read one with
`imageio.v3.imread(path)`, which returns `(K, H, W, 3)` uint8.

### `format: zarr`

```
export_zarr_<hz>hz/dataset.zarr     zarr format v2, the Diffusion Policy ReplayBuffer layout
  data/<key>                        all episodes concatenated along axis 0; images chunked per frame, Blosc zstd
  meta/episode_ends                 (E,) int64, exclusive end index of each episode
  .zattrs                           hz, pose_frame, prompts[E], episode_seeds[E], episode_names[E]
```

Episode `i` spans `episode_ends[i-1]:episode_ends[i]` (starting at 0). Images are stored losslessly.

### `format: oopsie` (`real_export` in `config/data.yaml`)

```
export_oopsie_<hz>hz/
  sim_<seed>.h5                   the real FR3 oopsiedata_format_v1 layout
  sim_<seed>_front.mp4            base_camera, <hz> fps
  sim_<seed>_wrist.mp4            wrist_camera
```

Same keys and conventions as the real recordings: `observations/robot_states/{joint_position,
joint_velocity, cartesian_position, gripper_position}`, `actions/{cartesian_position, gripper_position}`,
`observations/video_paths/{front,wrist}`, the `robot_profile` and `language_instruction` attributes and an
`episode_annotations/scripted` outcome of `success`. Poses are xyz + quaternion **xyzw** in `fr3_link0`, and
gripper values are the **total** finger opening (0 to 0.08 m). The real `diffusion_policy/dataset.py` and
`openpi_fr3/convert_oopsie_to_lerobot.py` read it unchanged.

Note: pi0/pi0.5 (openpi) train on LeRobot datasets, not zarr. Zarr here follows the Diffusion Policy
convention; converting either export to LeRobot needs a separate step.

## JSON contents

- `env_info`: `env_id`, `env_kwargs` (obs mode, control mode, backend) and `max_episode_steps`, enough to recreate the environment.
- `source_type` / `source_desc`: how the demos were produced.
- `episodes[i]`: `episode_id` (matches `traj_<id>`), `episode_seed` (resetting with this seed reproduces the initial scene), `elapsed_steps`, `reset_kwargs`, `success`, `fail`, and `prompt`, the language instruction sampled for that episode.

## Not saved

- The typed failure flags (`dropped`, `misaligned`, `stage`, `wrong_zone`, ...) and other per-step `info` values. Only `success` and `fail` reach the HDF5 file.
- Depth, segmentation and camera images beyond the two RGB views. Use `--obs-mode rgb+depth` or `rgb+segmentation` to add them.
- Distractor and prompt settings from the YAML at recording time; only the resulting prompt and actor states are recorded.
