# HiveBoard on ManiSkill

This project implements the 13 HiveBoard manipulation conditions in ManiSkill with a Franka Research 3 robot, a two-camera tabletop scene, task-specific success checks, and scripted demonstrations. It records successful trajectories and can export datasets for training. Robot and HiveBoard meshes are included locally.

[Setup](#setup) · [Record a demo](#record-a-demo) · [Cameras and randomization](#cameras-and-randomization) · [Tasks](#tasks) · [Website](#website) · [Citation](#citation) · [License](#license)

## Setup

Use Linux with a Vulkan-capable GPU driver. From the repository root, run:

```bash
make build
```

`make build` installs `uv`, Python 3.12, and the locked dependencies, then checks task registration and rendering. To open the basic FR3 table viewer, run `make run`.

## Record a demo

For one successful ball-valve trajectory and a side-by-side camera video:

```bash
uv run --locked python -m tasks.generate_demos --task ball_valve -n 1 --save-video
```

The recorder saves successful trajectories under `demos/<EnvId>/scripted_pd_ee_target_pose.{h5,json}`. With `--save-video`, it also writes `demos/<EnvId>/0.mp4`. Enabled exports are written in `export_<format>_<hz>hz/` under the same environment directory; their formats and rates are set in [`config/data.yaml`](config/data.yaml). See [`docs/DATASET.md`](docs/DATASET.md) for the recorded fields.

| Option | Meaning |
|---|---|
| `--task NAME` | Required; choose a name from [Tasks](#tasks). |
| `-n`, `--num-traj N` | Number of successful trajectories to save; default: `recording.num_episodes` in `config/data.yaml` (100). |
| `--record-dir PATH` | Output root; default: `demos`. |
| `--obs-mode MODE` | Recorded observation mode; default: `rgb`. Use `rgb+state` for object state or `rgb+depth` for depth. |
| `--save-video` | Save a side-by-side MP4 of the two task cameras for each successful trajectory. |
| `--showcase-video` | Also save an exterior-view showcase MP4 per attempt under `showcase/`. |
| `--image-hz N` | Keep camera frames at `N` Hz; `N` must divide the 20 Hz control rate. Incompatible exports are skipped. |
| `--viewer` | Show the scripted run in the SAPIEN viewer; uses one environment. |
| `--num-envs N` | Parallel simulated robots; default: 16. |
| `--seed N` | First episode seed; default: 0. |
| `--max-attempts N` | Maximum attempts to reach the requested number of successes; default: 1000. |
| `--sim-backend gpu\|cpu` | Simulation backend; default: `gpu`. |
| `--domain-randomization` / `--no-domain-randomization` | Enable or disable the configured robot, camera, and visual variation for this run; default: disabled. |

The shorter command `make demos ball_valve N=1` uses the same recorder. Use `make video ball_valve` for one video, or `make view ball_valve` to watch a scripted run live.

## Cameras and randomization

Edit the selected task's configuration, for example [`tasks/ball_valve/config.yaml`](tasks/ball_valve/config.yaml), before recording. `cameras.base_camera` sets the fixed view (`eye`, `target`, field of view); `cameras.wrist_camera` sets the hand-mounted view (`position`, quaternion, field of view). `cameras.resolution` sets both image sizes. The same file controls board and object placement, robot start sampling under `randomization`, distractors, and prompt templates.

[`config/config.yaml`](config/config.yaml) contains global camera-pose ranges, lighting, color, texture, image variation, robot response, and camera latency. Set `domain_randomization: true` there, or pass `--domain-randomization` when recording, to enable those effects. Task placement sampling remains controlled by each task file. Use `--seed` to reproduce a sampled start.

## Tasks

Each task mounts a HiveBoard mechanism or loose part on the board. The scripted controller acts on it, and the environment checks whether the required end state is reached. The available task names are `ball_valve`, `ball_valve_ring`, `circuit_breaker`, `gate_valve_small`, `gate_valve_large`, `button_cover`, `key_lock`, `drawer`, `peg_insertion`, `thread_m8`, `thread_m30`, `light_bulb`, and `shock_absorber`. For each task's starting state, action, and success criterion, see the [HiveBoard task guide](https://hiveboard-bench.github.io/hivedocs/benchmark/tasks).

## Website

[HiveBoard project website](https://hiveboard-bench.github.io/) · [Documentation](https://hiveboard-bench.github.io/hivedocs/)

## Citation

If you use HiveBoard in research, cite the [project paper](https://hiveboard-bench.github.io/hivedocs/reference/citation):

```bibtex
@article{hiveboard2026,
  title   = {HiveBoard: An Open, Modular, 3D-Printed Benchmark of Industrial Mechanisms for Robotic and Prosthetic Manipulation},
  author  = {Godoy, Ricardo V. and de Souza, Enzo F. and Angarola, Matheus P. and de Lange, Rudy De-Xin and Negri, Juliano and Marsicano, Joao A. and others},
  journal = {Under review},
  year    = {2026},
  url     = {https://github.com/hiveboard-bench/HiveBoard}
}
```

## License

The code is MIT licensed under the repository [LICENSE](LICENSE). The included [HiveBoard meshes](fr3_assets/hiveboard/LICENSE) are MIT licensed, and the [FR3 assets](fr3_assets/LICENSE) are Apache 2.0 licensed.
