# Previews camera poses in the simulation.
# Created: 2026-10-16

from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v3 as iio
import numpy as np
import torch
from PIL import Image, ImageDraw

from env import fr3_table_env
from mani_skill.utils import sapien_utils
from tasks.circuit_breaker.task import CircuitBreakerFR3Env


# Convert a rendered image to RGB.
def rgb(sensor_data):
    image = sensor_data["rgb"][0].cpu().numpy()
    if np.issubdtype(image.dtype, np.floating):
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
    return image


def main():
    root = Path("outputs/camera_calibration")
    root.mkdir(parents=True, exist_ok=True)
    cameras = CircuitBreakerFR3Env.config["cameras"]
    cameras["resolution"] = {"width": 640, "height": 480}
    cameras["base_camera"]["fov_degrees"] = 70.0
    cameras["wrist_camera"]["fov_degrees"] = 42.5

    env = gym.make("CircuitBreakerFR3-v1", obs_mode="rgb", render_mode="rgb_array")
    try:
        env.reset(seed=0)
        with h5py.File(
            "/home/math/oopsie_dataset/samples/20260917_133900/20260917_133952.h5"
        ) as record:
            joints = record["observations/robot_states/joint_position"][30]
        qpos = env.unwrapped.agent.robot.get_qpos()
        qpos[0, :7] = torch.as_tensor(joints, dtype=qpos.dtype, device=qpos.device)
        env.unwrapped.agent.robot.set_qpos(qpos)
        initial = env.unwrapped._get_obs_sensor_data()
        iio.imwrite(root / "front_sim.png", rgb(initial["base_camera"]))
        iio.imwrite(root / "wrist_sim.png", rgb(initial["wrist_camera"]))
        sensor = env.unwrapped._sensors["base_camera"].camera
        target = [0.20, 0.42, 0.25]
        eyes = [
            [0.35, -0.10, 0.55], [0.45, -0.10, 0.55], [0.55, -0.10, 0.55],
            [0.35, -0.10, 0.65], [0.45, -0.10, 0.65], [0.55, -0.10, 0.65],
        ]
        sheet = Image.new("RGB", (640 * 3, 510 * 2), "white")
        draw = ImageDraw.Draw(sheet)
        for index, eye in enumerate(eyes):
            sensor.set_local_pose(sapien_utils.look_at(eye=eye, target=target).sp)
            frame = rgb(env.unwrapped._get_obs_sensor_data()["base_camera"])
            iio.imwrite(root / f"front_candidate_{index}.png", frame)
            x, y = (index % 3) * 640, (index // 3) * 510
            sheet.paste(Image.fromarray(frame), (x, y + 30))
            draw.text((x + 10, y + 8), f"{index}: eye {eye}", fill="black")
        sheet.save(root / "front_candidates.png")
        wrist_sensor = env.unwrapped._sensors["wrist_camera"].camera
        hand = env.unwrapped.agent.hand.pose.sp
        wrist_sheet = Image.new("RGB", (640 * 3, 510), "white")
        wrist_draw = ImageDraw.Draw(wrist_sheet)
        for index, target_y in enumerate((0.47, 0.49, 0.51)):
            eye = hand.p + np.array([0.14, 0.0, -0.06])
            target_pose = sapien_utils.look_at(
                eye=eye, target=[0.11, target_y, 0.0], up=[0, 1, 0]
            ).sp
            local_pose = hand.inv() * target_pose
            wrist_sensor.set_local_pose(local_pose)
            frame = rgb(env.unwrapped._get_obs_sensor_data()["wrist_camera"])
            iio.imwrite(root / f"wrist_candidate_{index}.png", frame)
            wrist_sheet.paste(Image.fromarray(frame), (index * 640, 30))
            wrist_draw.text((index * 640 + 10, 8), f"target y={target_y}", fill="black")
        wrist_sheet.save(root / "wrist_candidates.png")
    finally:
        env.close()


if __name__ == "__main__":
    main()
