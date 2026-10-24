# HiveBoard assets

Meshes from [hiveboard-bench/HiveBoard](https://github.com/hiveboard-bench/HiveBoard) (`Simulation/`, see `LICENSE`).
The `.mtl` references were stripped; tasks set the colours. Part placements follow the `Simulation/*/*.urdf` origins,
copied into each task's `config.yaml` (`assembly` section, see `tasks/hiveboard_assembly.py`).

| Folder | HiveBoard source |
|---|---|
| `drawer/` | `Simulation/Drawer` |
| `peg_insertion/` | `Simulation/Peg Insertion` |
| `thread_m8/` | `Simulation/Threads/M8 Thread` |
| `thread_m30/` | `Simulation/Threads/M30 Thread` |
| `light_bulb/` | `Simulation/Lamp` |
| `shock_absorber/` | `Simulation/Shock Absorber` (only the shock absorber; `tasks/shock_absorber` mounts it on the Thread M8 and Peg insertion bases) |

`*_coacd.obj` are CoACD convex decompositions (one object per convex piece) of meshes whose holes and recesses matter
for collisions: the nuts, the bulb socket, the peg plate and the drawer. A single convex hull would fill them. `light_bulb/bulb_coacd.obj` decomposes the whole bulb (thread and globe, in its URDF link frame), so it
can lie on the table without rolling. `shock_absorber/Corpo3_seg0..4.obj` cut the shock absorber into its rigid ends
and three coil segments (with their `_coacd.obj`), which `tasks/shock_absorber` joins with springs. Regenerate with `scripts/decompose_hiveboard_meshes.py`.
