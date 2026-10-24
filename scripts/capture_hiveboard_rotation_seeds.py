# Captures HiveBoard rotation seeds.
# Created: 2026-10-16

from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from mani_skill.utils import sapien_utils
from tasks.circuit_breaker.task import CircuitBreakerFR3Env


def main():
    output = Path("outputs/hiveboard_rotation_seeds")
    output.mkdir(parents=True, exist_ok=True)
    env = gym.make("CircuitBreakerFR3-v1", obs_mode="rgb", render_mode="rgb_array")
    sheet = Image.new("RGB", (640 * 5, 510 * 2), "white")
    draw = ImageDraw.Draw(sheet)
    overhead = sapien_utils.look_at(
        eye=[0.0, 0.475, 0.48], target=[0.0, 0.475, 0.0], up=[0, 1, 0]
    ).sp
    try:
        for seed in range(10):
            env.reset(seed=seed)
            base = env.unwrapped
            base._sensors["base_camera"].camera.set_local_pose(overhead)
            frame = base._get_obs_sensor_data()["base_camera"]["rgb"][0].cpu().numpy()
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            iio.imwrite(output / f"seed_{seed}.png", frame)
            x, y = (seed % 5) * 640, (seed // 5) * 510
            sheet.paste(Image.fromarray(frame), (x, y + 30))
            angle = int(base.module_yaw_degrees[0].item())
            draw.text((x + 12, y + 8), f"seed {seed} | modulo {angle:+d} deg", fill="black")
        sheet.save(output / "comparison.png")
    finally:
        env.close()


if __name__ == "__main__":
    main()
