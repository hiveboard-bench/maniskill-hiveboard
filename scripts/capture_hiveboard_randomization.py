# Captures randomized HiveBoard scenes.
# Created: 2026-10-22

import argparse
import json
from collections import Counter
from pathlib import Path

import gymnasium as gym
import numpy as np
import sapien

import tasks
from mani_skill.utils import sapien_utils
from scripts.capture_hiveboard_tasks import loose_part, render
from tasks.generate_demos import CONTROL_MODE, CartesianPlanner


# Show or hide the arm, which can sit between the overhead camera and the board.
def set_robot_visible(base, visible):
    for link in base.agent.robot.links:
        for obj in link._objs:
            body = obj.entity.find_component_by_type(sapien.render.RenderBodyComponent)
            if body is not None:
                body.visibility = 1.0 if visible else 0.0


# (cell index, in-cell rotation in degrees) of env 0, the rotation without the board yaw.
def placement(base):
    if hasattr(base, "peg_cell_index"):
        return int(base.peg_cell_index[0].item()) + 1, int(base.peg_hole_index[0].item())
    offset = base.config["objects"].get("module_yaw_offset_degrees", 0.0)
    rotation = (base.module_yaw_degrees[0] - base.board_yaw_degrees[0]).item() + offset
    return int(base.module_cell_index[0].item()), int(round(rotation)) % 360


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=sorted(tasks.TASKS))
    parser.add_argument("--seeds", type=int, default=12, help="seeds photographed per task, from 0")
    parser.add_argument("--count-seeds", type=int, default=300, help="seeds whose placement is only counted")
    parser.add_argument("--settle-steps", type=int, default=20)
    parser.add_argument("--output", type=Path, default=Path("outputs/hiveboard_randomization"))
    parser.add_argument("--size", type=int, nargs=2, default=(480, 360))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    summary_path = args.output / "summary.json"
    summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
    for task in args.tasks:
        tasks.ENV_CLASSES[task].config["cameras"]["resolution"] = {"width": args.size[0], "height": args.size[1]}
        env = gym.make(tasks.TASKS[task], obs_mode="rgb", control_mode=CONTROL_MODE, sim_backend="cpu")
        base = env.unwrapped
        shots = []
        try:
            for seed in range(args.seeds):
                env.reset(seed=seed)
                cell, rotation = placement(base)
                part = loose_part(base)
                start = part.pose.p[0].cpu().numpy().copy() if part is not None else None
                CartesianPlanner(env).idle(args.settle_steps)
                moved = float(np.linalg.norm(part.pose.p[0].cpu().numpy() - start) * 1000) if part is not None else None
                board = base.board.pose.p[0].cpu().numpy()
                height = 0.25 / np.tan(np.deg2rad(base.config["cameras"]["base_camera"]["fov_degrees"]) / 2)
                overhead = sapien_utils.look_at(eye=[board[0], board[1] - 0.001, board[2] + height], target=board.tolist()).sp
                path = args.output / f"{task}_seed{seed}.jpg"
                set_robot_visible(base, False)
                render(base, overhead).save(path, quality=85)
                set_robot_visible(base, True)
                shots.append(dict(seed=seed, cell=cell, rotation=rotation, image=path.name,
                                  part_moved_mm=None if moved is None else round(moved, 2)))
                print(f"{task} seed={seed} cell={cell} rotation={rotation} part moved {moved} mm")
            counts = Counter()
            for seed in range(args.count_seeds):
                env.reset(seed=seed)
                counts[placement(base)] += 1
        finally:
            env.close()
        cells = base.config["objects"]["module_cell_positions_xy"]
        summary[task] = dict(
            shots=shots,
            count_seeds=args.count_seeds,
            counts={f"{cell}:{rotation}": n for (cell, rotation), n in sorted(counts.items())},
            cells=len(cells),
            rotations=sorted({rotation for _, rotation in counts}),
        )
        print(f"{task}: {len(counts)} cell/rotation combinations in {args.count_seeds} seeds")
    summary_path.write_text(json.dumps(summary, indent=1))


if __name__ == "__main__":
    main()
