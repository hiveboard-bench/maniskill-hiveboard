# Captures images of HiveBoard tasks.
# Created: 2026-10-22

import argparse
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.spatial.transform import Rotation

import tasks
from mani_skill.utils import sapien_utils
from tasks.generate_demos import CONTROL_MODE, CartesianPlanner

VIEWS = ("overview", "module", "part", "base_camera")


# Render the current scene.
def render(base, pose=None):
    camera = base._sensors["base_camera"].camera
    if pose is not None:
        camera.set_local_pose(pose)
    frame = base._get_obs_sensor_data()["base_camera"]["rgb"][0].cpu().numpy()
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return Image.fromarray(frame)


# Position the camera for a close view of the task module.
def close_up(target, distance, azimuth_degrees, elevation_degrees):
    azimuth, elevation = np.deg2rad(azimuth_degrees), np.deg2rad(elevation_degrees)
    offset = distance * np.array([np.cos(elevation) * np.cos(azimuth), np.cos(elevation) * np.sin(azimuth), np.sin(elevation)])
    return sapien_utils.look_at(eye=(target + offset).tolist(), target=target.tolist()).sp


# Return the loose task part.
def loose_part(base):
    for name in ("part", "key", "shock"):
        if hasattr(base, name):
            return getattr(base, name)
    return None


# Capture the current frame.
def capture(task, seed, settle_steps, size):
    env_class = tasks.ENV_CLASSES[task]
    env_class.config["cameras"]["resolution"] = {"width": size[0], "height": size[1]}
    env = gym.make(tasks.TASKS[task], obs_mode="rgb", control_mode=CONTROL_MODE, sim_backend="cpu")
    try:
        env.reset(seed=seed)
        base = env.unwrapped
        part = loose_part(base)
        start = part.pose.raw_pose[0].cpu().numpy().copy() if part is not None else None
        CartesianPlanner(env).idle(settle_steps)
        report = ""
        if part is not None:
            end = part.pose.raw_pose[0].cpu().numpy()
            moved = np.linalg.norm(end[:3] - start[:3]) * 1000
            turned = np.rad2deg((Rotation.from_quat(end[[4, 5, 6, 3]]) * Rotation.from_quat(start[[4, 5, 6, 3]]).inv()).magnitude())
            speed = np.linalg.norm(part.linear_velocity[0].cpu().numpy()) * 1000
            report = f"part moved {moved:.2f} mm, turned {turned:.2f} deg, speed {speed:.2f} mm/s after {settle_steps} steps"
        board = base.board.pose.p[0].cpu().numpy()
        module = base.valve.pose.p[0].cpu().numpy()
        views = {
            "overview": sapien_utils.look_at(eye=[board[0], board[1] - 0.001, board[2] + 0.55], target=board.tolist()).sp,
            "module": close_up(module + [0, 0, 0.02], 0.30, -90 + 35, 35),
            "part": close_up((part.pose.p[0].cpu().numpy() if part is not None else module) + [0, 0, 0.015], 0.22, -90 - 35, 30),
            "base_camera": None,
        }
        base_pose = base._sensors["base_camera"].camera.get_local_pose().sp
        images = {name: render(base, pose if pose is not None else base_pose) for name, pose in views.items()}
    finally:
        env.close()
    return images, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", nargs="+", default=sorted(tasks.TASKS))
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--settle-steps", type=int, default=60, help="20 Hz control steps holding the arm still")
    parser.add_argument("--output", type=Path, default=Path("outputs/hiveboard_tasks"))
    parser.add_argument("--size", type=int, nargs=2, default=(640, 480))
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    width, height = args.size
    for task in args.tasks:
        for seed in args.seeds:
            images, report = capture(task, seed, args.settle_steps, args.size)
            sheet = Image.new("RGB", (width * 2, height * 2 + 30), "white")
            for index, name in enumerate(VIEWS):
                sheet.paste(images[name], ((index % 2) * width, 30 + (index // 2) * height))
                images[name].save(args.output / f"{task}_seed{seed}_{name}.png")
            ImageDraw.Draw(sheet).text((10, 8), f"{task} seed {seed} | {report}", fill="black")
            sheet.save(args.output / f"{task}_seed{seed}.png")
            print(f"{task} seed={seed} {report} -> {args.output / f'{task}_seed{seed}.png'}")
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
