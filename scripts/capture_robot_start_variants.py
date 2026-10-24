# Captures robot start pose variants.
# Created: 2026-10-16

from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from mani_skill.utils import sapien_utils
from tasks.circuit_breaker.task import CircuitBreakerFR3Env


def main():
    output = Path("outputs/robot_start_variants")
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("CircuitBreakerFR3-v1", obs_mode="rgb", render_mode="rgb_array")
    sheet = Image.new("RGB", (1280, 1020), "white")
    draw = ImageDraw.Draw(sheet)
    try:
        overview_pose = sapien_utils.look_at(
            eye=[0.85, -0.95, 1.0], target=[0.0, 0.20, 0.35]
        ).sp
        seeds = (0, 1, 2, 3)
        poses = []
        for seed in seeds:
            env.reset(seed=seed)
            poses.append(env.unwrapped.agent.robot.get_qpos().clone())
        env.reset(seed=0)
        env.unwrapped._sensors["base_camera"].camera.set_local_pose(overview_pose)
        for index, (seed, qpos) in enumerate(zip(seeds, poses)):
            env.unwrapped.agent.robot.set_qpos(qpos)
            frame = env.unwrapped._get_obs_sensor_data()["base_camera"]["rgb"][0]
            frame = frame.cpu().numpy()
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            iio.imwrite(output / f"overview_seed_{seed}.png", frame)
            x, y = (index % 2) * 640, (index // 2) * 510
            sheet.paste(Image.fromarray(frame), (x, y + 30))
            gripper = "aberto" if qpos[0, 7].item() > 0.02 else "fechado"
            draw.text((x + 12, y + 8), f"seed {seed} - gripper {gripper}", fill="black")
        sheet.save(output / "overview_comparison.png")
    finally:
        env.close()


if __name__ == "__main__":
    main()
