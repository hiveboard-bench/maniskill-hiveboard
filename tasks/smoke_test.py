# Checks task registration, episode execution, and scripted failure conditions.
# Created: 2026-09-14

import argparse
import sys

import gymnasium as gym
import numpy as np

from tasks import ENV_CLASSES, TASKS
from tasks.generate_demos import CLOSED, CONTROL_MODE, SOLVERS, CartesianPlanner, make_pose, rotation, top_down, wrap


# Count episode outcomes and check expected failure flags.
def summarize(info, keys):
    return {key: round(float(info[key].item()), 3) for key in keys}


# Parse command line options and run the requested task workflow.
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", choices=TASKS, required=True)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--sim-backend", choices=("cpu", "gpu"), default="cpu")
    parser.add_argument("--distractors", action="store_true", help="enable every distractor type, ignoring the yaml flags")
    args = parser.parse_args()
    if args.distractors:
        for kind in ("cubes", "ood_objects", "wrong_zones"):
            ENV_CLASSES[args.task].config["distractors"][kind]["enabled"] = True
        if "ycb" in ENV_CLASSES[args.task].config["distractors"]:
            ENV_CLASSES[args.task].config["distractors"]["ycb"]["enabled"] = True

    env = gym.make(
        TASKS[args.task], obs_mode="rgb", control_mode=CONTROL_MODE, sim_backend=args.sim_backend
    )
    base = env.unwrapped
    keys = base.info_keys
    obs, _ = env.reset(seed=0)
    cameras = {name: tuple(data["rgb"].shape) for name, data in obs["sensor_data"].items()}
    print(f"{TASKS[args.task]} action_space={env.action_space.shape} cameras={cameras}")
    failed = sorted(cameras) != ["base_camera", "wrist_camera"]

    for seed in range(args.episodes):
        env.reset(seed=seed)
        done = False
        while not done:
            _, _, terminated, truncated, info = env.step(env.action_space.sample())
            done = bool(terminated.any() or truncated.any())
        print(f"random seed={seed}", summarize(info, keys))

    scenarios = [(None, ["success"])]
    scenarios += [(key, [key]) for key in base.failure_keys if key != "timeout"]
    for fault, expected in scenarios:
        if fault and fault.startswith("never_"):
            expected.append("timeout")
            break
    for index, (fault, expected) in enumerate(scenarios):
        for seed in range(index * 10, index * 10 + 3):
            env.reset(seed=seed)
            info = SOLVERS[args.task](CartesianPlanner(env), fault)
            ok = all(bool(info[key].item()) for key in expected)
            if ok:
                break
        failed |= not ok
        print(f"scripted fault={fault} seed={seed} expect={expected} {'PASS' if ok else 'FAIL'}", summarize(info, keys))

    if base.distractors:
        env.reset(seed=100)
        planner = CartesianPlanner(env)
        actor = base.distractors[0][0]
        target = actor.pose.sp
        grasp = top_down(wrap(rotation(target).as_euler("xyz")[2], np.pi / 2))
        planner.move_to(make_pose(target.p + [0, 0, 0.08], grasp))
        planner.move_to(make_pose(target.p, grasp), pos_tol=0.002)
        planner.set_gripper(CLOSED)
        info = planner.info
        ok = bool(info["grasped_distractor"].item())
        failed |= not ok
        print(f"scripted grasp {actor.name} expect=['grasped_distractor'] {'PASS' if ok else 'FAIL'}", summarize(info, keys))

    env.close()
    print("FAILED" if failed else "ALL CHECKS PASSED")
    sys.exit(int(failed))


if __name__ == "__main__":
    main()
