# Runs a scripted FR3 task and records its camera views as a video.
# Created: 2026-10-16

import argparse
from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
from PIL import Image

import tasks  # noqa: F401 - registers the custom environments
from tasks.generate_demos import CONTROL_MODE, CartesianPlanner, SOLVERS


# Parse command line options and run the requested task workflow.
def main():
    parser = argparse.ArgumentParser(description="Record a scripted FR3 task video")
    parser.add_argument("--task", choices=sorted(SOLVERS), required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("task_video.mp4"))
    parser.add_argument("--orbit", action="store_true", help="orbit the exterior camera")
    parser.add_argument("--orbit-radius", type=float, default=1.25)
    parser.add_argument("--orbit-height", type=float, default=0.72)
    parser.add_argument("--orbit-degrees", type=float, default=60.0)
    parser.add_argument("--orbit-frames", type=int, default=900,
                        help="frames over which the orbit-degrees sweep occurs")
    parser.add_argument("--width", type=int, default=1280, help="real camera width (HD default)")
    parser.add_argument("--height", type=int, default=720, help="real camera height")
    parser.add_argument("--scale", type=float, default=1.0,
                        help="additional output scale after camera capture")
    parser.add_argument("--fps", type=int, default=30)
    args = parser.parse_args()

    env_id = tasks.TASKS[args.task]
    tasks.ENV_CLASSES[args.task].config["cameras"]["resolution"] = {
        "width": args.width, "height": args.height
    }
    env = gym.make(
        env_id,
        num_envs=1,
        obs_mode="rgb+state",
        control_mode=CONTROL_MODE,
        render_mode="sensors",
        sim_backend="cpu",
    )
    base = env.unwrapped
    wrist_position = base.config["cameras"]["wrist_camera"]["position"]
    camera_target = np.asarray(base.config["cameras"]["base_camera"]["target"], dtype=float)
    frames = []
    frame_index = 0

    # Move the exterior camera along the configured orbit.
    def orbit_camera():
        nonlocal frame_index
        if not args.orbit:
            return
        angle = np.deg2rad(args.orbit_degrees) * frame_index / max(args.orbit_frames, 1)
        center = camera_target.copy()
        center[2] = 0.0
        eye = center + np.array([
            args.orbit_radius * np.cos(angle),
            args.orbit_radius * np.sin(angle),
            args.orbit_height,
        ])
        base.set_camera_positions(eye.tolist(), wrist_position)

    # Render camera feeds and append the composed video frame.
    def capture(obs):
        nonlocal frame_index
        rgb = obs["sensor_data"]["base_camera"]["rgb"][0]
        if hasattr(rgb, "detach"):
            rgb = rgb.detach().cpu().numpy()
        frame = np.asarray(rgb).copy()
        if args.scale != 1.0:
            height, width = frame.shape[:2]
            frame = np.asarray(Image.fromarray(frame).resize(
                (round(width * args.scale), round(height * args.scale)), Image.Resampling.LANCZOS
            ))
        frames.append(frame)
        frame_index += 1
        orbit_camera()

    try:
        obs, _ = env.reset(seed=args.seed)
        capture(obs)

        # Apply the scripted action and capture the resulting frame.
        def step(action):
            obs, _, terminated, truncated, info = env.step(action)
            capture(obs)
            return terminated, truncated, info

        planner = CartesianPlanner(env, step=step, info=base.get_info())
        info = SOLVERS[args.task](planner)
        success = bool(info["success"].item())
        args.output.parent.mkdir(parents=True, exist_ok=True)
        iio.imwrite(args.output, np.stack(frames), fps=args.fps, codec="libx264")
        print(f"saved {len(frames)} frames to {args.output} (success={success})")
    finally:
        env.close()


if __name__ == "__main__":
    main()
