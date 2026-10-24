# Captures images with different camera settings.
# Created: 2026-10-16

from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw

from tasks.circuit_breaker.task import CircuitBreakerFR3Env


# Convert a rendered image to RGB.
def rgb(data):
    frame = data["rgb"][0].cpu().numpy()
    if np.issubdtype(frame.dtype, np.floating):
        frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
    return frame


def main():
    output = Path("outputs/camera_variants")
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("CircuitBreakerFR3-v1", obs_mode="rgb", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        with h5py.File(
            "/home/math/oopsie_dataset/samples/20260917_133900/20260917_133952.h5"
        ) as record:
            joints = record["observations/robot_states/joint_position"][30]
        robot = env.unwrapped.agent.robot
        qpos = robot.get_qpos()
        qpos[0, :7] = torch.as_tensor(joints, dtype=qpos.dtype, device=qpos.device)
        robot.set_qpos(qpos)

        sheets = {
            name: Image.new("RGB", (1280, 1020), "white")
            for name in ("base_camera", "wrist_camera")
        }
        for index, seed in enumerate((0, 1, 2, 3)):
            env.unwrapped._episode_seed[0] = seed
            env.unwrapped._set_randomized_cameras(torch.tensor([0]))
            observations = env.unwrapped._get_obs_sensor_data()
            for name, sheet in sheets.items():
                frame = rgb(observations[name])
                iio.imwrite(output / f"{name}_seed_{seed}.png", frame)
                x, y = (index % 2) * 640, (index // 2) * 510
                sheet.paste(Image.fromarray(frame), (x, y + 30))
                ImageDraw.Draw(sheet).text((x + 12, y + 9), f"seed {seed}", fill="black")
        for name, sheet in sheets.items():
            sheet.save(output / f"{name}_comparison.png")
    finally:
        env.close()


if __name__ == "__main__":
    main()
