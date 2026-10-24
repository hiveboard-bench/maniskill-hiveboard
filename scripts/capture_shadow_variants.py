# Captures scenes with different shadows.
# Created: 2026-10-16

from pathlib import Path

import gymnasium as gym
import imageio.v3 as iio
import numpy as np
from PIL import Image, ImageDraw

from env import fr3_table_env
from mani_skill.utils import sapien_utils


def main():
    output_dir = Path("outputs/shadow_variants")
    output_dir.mkdir(parents=True, exist_ok=True)

    front = fr3_table_env.CONFIG["cameras"]["front"]
    front["width"], front["height"] = 960, 720
    env = gym.make("FR3Table-v1", obs_mode="rgb", render_mode="rgb_array")
    frames = []
    try:
        for seed in (0, 1, 2, 3):
            env.reset(seed=seed)
            pose = sapien_utils.look_at(eye=[1.35, -2.2, 1.65], target=[0, 0, 0.05]).sp
            env.unwrapped._sensors["front_camera"].camera.set_local_pose(pose)
            rgb = env.unwrapped._get_obs_sensor_data()["front_camera"]["rgb"][0]
            frame = rgb.cpu().numpy()
            if np.issubdtype(frame.dtype, np.floating):
                frame = np.clip(frame * 255, 0, 255).astype(np.uint8)
            iio.imwrite(output_dir / f"seed-{seed}.png", frame)
            frames.append(Image.fromarray(frame))
            print(seed, env.unwrapped.lighting_parameters)
    finally:
        env.close()

    contact = Image.new("RGB", (1920, 1500), "#eeeeee")
    draw = ImageDraw.Draw(contact)
    for index, frame in enumerate(frames):
        x, y = (index % 2) * 960, (index // 2) * 750
        contact.paste(frame, (x, y + 30))
        draw.text((x + 12, y + 8), f"Seed {index}", fill="black")
    contact.save(output_dir / "comparacao.png")
    print(f"Saved {output_dir / 'comparacao.png'}")


if __name__ == "__main__":
    main()
