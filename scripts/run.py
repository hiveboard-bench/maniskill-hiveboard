# Runs the interactive FR3 simulation.
# Created: 2026-09-14

import argparse
import itertools
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import yaml

from env import fr3_table_env


class SliderPanel:
    """Draw labeled sliders in one OpenCV window."""

    def __init__(self, cv2, window, controls, camera_height=0):
        self.cv2 = cv2
        self.window = window
        self.controls = controls
        self.camera_height = camera_height
        self.width = 760
        self.row_height = 58
        self.controls_top = 12 + camera_height
        self.height = self.controls_top + len(controls) * self.row_height + 12
        self.dragging = None
        self.bars = []
        cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE | cv2.WINDOW_GUI_NORMAL)
        cv2.setMouseCallback(window, self._mouse)

    # Update a slider from the mouse position.
    def _set_from_mouse(self, index, x):
        left, right, _, _ = self.bars[index]
        ratio = np.clip((x - left) / (right - left), 0.0, 1.0)
        lower, upper = self.controls[index]["limits"]
        self.controls[index]["value"] = lower + ratio * (upper - lower)

    # Handle mouse events for the slider panel.
    def _mouse(self, event, x, y, flags, _):
        if event == self.cv2.EVENT_LBUTTONDOWN:
            for index, (left, right, top, bottom) in enumerate(self.bars):
                if left - 12 <= x <= right + 12 and top <= y <= bottom:
                    self.dragging = index
                    self._set_from_mouse(index, x)
                    break
        elif event == self.cv2.EVENT_MOUSEMOVE and self.dragging is not None:
            self._set_from_mouse(self.dragging, x)
        elif event == self.cv2.EVENT_LBUTTONUP:
            self.dragging = None

    # Return the current slider values.
    def values(self):
        return [control["value"] for control in self.controls]

    # Draw the slider panel.
    def draw(self, camera_image=None):
        cv2 = self.cv2
        canvas = np.full((self.height, self.width, 3), (242, 242, 242), np.uint8)

        if camera_image is not None:
            image = cv2.resize(camera_image, (740, self.camera_height - 20))
            canvas[10:10 + image.shape[0], 10:750] = image
            cv2.rectangle(canvas, (20, 20), (141, 48), (65, 65, 65), -1)
            cv2.rectangle(canvas, (389, 20), (540, 48), (65, 65, 65), -1)
            cv2.putText(canvas, "FRONT CAMERA", (30, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)
            cv2.putText(canvas, "GRIPPER CAMERA", (399, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (235, 235, 235), 1)

        self.bars = []
        for index, control in enumerate(self.controls):
            y = self.controls_top + index * self.row_height
            lower, upper = control["limits"]
            ratio = (control["value"] - lower) / (upper - lower)
            left, right, track_y = 202, 724, y + 20
            color = (92, 92, 92)
            knob_x = round(left + np.clip(ratio, 0.0, 1.0) * (right - left))
            cv2.putText(canvas, control["label"], (24, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (48, 48, 48), 1)
            cv2.putText(canvas, f'{control["value"]:+.4f}', (112, y + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
            cv2.line(canvas, (left, track_y), (right, track_y), (190, 190, 190), 6, cv2.LINE_AA)
            cv2.line(canvas, (left, track_y), (knob_x, track_y), color, 6, cv2.LINE_AA)
            cv2.circle(canvas, (knob_x, track_y), 9, (255, 255, 255), -1, cv2.LINE_AA)
            cv2.circle(canvas, (knob_x, track_y), 5, color, -1, cv2.LINE_AA)
            cv2.putText(canvas, f"{lower:+.2f}", (left, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (125, 125, 125), 1)
            cv2.putText(canvas, f"{upper:+.2f}", (right - 38, y + 45), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (125, 125, 125), 1)
            self.bars.append((left, right, y, y + 42))
        return canvas


# Return a frame combining the selected camera feeds.
def camera_frame(env, names=("front_camera", "gripper_camera")):
    images = env.unwrapped._get_obs_sensor_data()
    frame = torch.cat(
        [images[names[0]]["rgb"], images[names[1]]["rgb"]], dim=2
    )
    if frame.dtype.is_floating_point:
        frame = torch.clip(frame * 255, 0, 255).to(torch.uint8)
    env.unwrapped.show_camera_markers()
    return frame


# Compute the camera pose from the controls.
def front_camera_from_controls(eye, target, values):
    eye = np.asarray(eye, dtype=float)
    target = np.asarray(target, dtype=float)
    direction = target - eye
    distance = np.linalg.norm(direction)
    forward = direction / distance
    horizontal = np.array([forward[0], forward[1], 0.0])
    horizontal /= np.linalg.norm(horizontal)
    right = np.cross(horizontal, [0.0, 0.0, 1.0])
    side, depth, height, yaw, pitch = values
    offset = side * right + depth * horizontal + np.array([0.0, 0.0, height])
    new_eye = eye + offset
    heading = np.arctan2(forward[1], forward[0]) + np.deg2rad(yaw)
    elevation = np.arcsin(forward[2]) + np.deg2rad(pitch)
    elevation = np.clip(elevation, np.deg2rad(-85), np.deg2rad(85))
    new_direction = np.array([
        np.cos(elevation) * np.cos(heading),
        np.cos(elevation) * np.sin(heading),
        np.sin(elevation),
    ])
    return new_eye.tolist(), (new_eye + distance * new_direction).tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--camera-gui", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--joint-gui", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--task", default=None)
    args = parser.parse_args()

    config = fr3_table_env.CONFIG
    simulation = config["simulation"]
    camera_names = ("front_camera", "gripper_camera")
    if args.task:
        from tasks import TASKS, ENV_CLASSES
        if args.task not in TASKS:
            parser.error(f"unknown task: {args.task}")
        config = ENV_CLASSES[args.task].config
        simulation = {"obs_mode": "rgb", "control_mode": "pd_ee_delta_pose", "sim_backend": "cpu", "seed": 0, "num_envs": 1, "steps": 0}
        camera_names = ("base_camera", "wrist_camera")
    gui_config = config.get("camera_gui", fr3_table_env.CONFIG["camera_gui"])
    joint_config = config.get("joint_gui", fr3_table_env.CONFIG["joint_gui"])
    camera_gui = gui_config.get("enabled", False) if args.camera_gui is None else args.camera_gui
    joint_gui = joint_config.get("enabled", False) if args.joint_gui is None else args.joint_gui
    if args.headless and (camera_gui or joint_gui):
        parser.error("GUI flags cannot be used with --headless")
    if joint_gui and simulation["num_envs"] != 1:
        parser.error("joint control requires num_envs: 1")
    if joint_gui and simulation["control_mode"] != "pd_joint_pos":
        parser.error("joint control requires control_mode: pd_joint_pos")

    env = gym.make(
        TASKS[args.task] if args.task else simulation["env_id"],
        obs_mode=simulation["obs_mode"],
        control_mode=simulation["control_mode"],
        render_mode="sensors",
        sim_backend=simulation["sim_backend"],
        num_envs=simulation["num_envs"],
    )
    env.reset(seed=simulation["seed"])
    rest = np.asarray(fr3_table_env.FR3.keyframes["rest"].qpos, dtype=np.float32)
    action = np.zeros((1, 7), dtype=np.float32) if args.task else np.concatenate([rest[:7], rest[7:8]])[None, :]

    cv2 = None
    camera_panel = None
    joint_panel = None
    if not args.headless and (camera_gui or joint_gui):
        import cv2

    if camera_gui:
        front_name = "base_camera" if args.task else "front"
        wrist_name = "wrist_camera" if args.task else "gripper"
        front_initial_eye = list(config["cameras"][front_name]["eye"])
        front_initial_target = list(config["cameras"][front_name]["target"])
        controls = [
            {"label": label, "value": 0.0, "limits": limits}
            for label, limits in (
                ("Side (m)", (-1.5, 1.5)),
                ("Near (m)", (-1.5, 1.5)),
                ("Height (m)", (-1.0, 1.0)),
                ("Pan (deg)", (-90.0, 90.0)),
                ("Tilt (deg)", (-60.0, 60.0)),
            )
        ]
        camera_limits = config.get("camera_gui", fr3_table_env.CONFIG["camera_gui"])
        for index, axis in enumerate("xyz"):
            controls.append({
                "label": f"Wrist {axis.upper()}",
                "value": config["cameras"][wrist_name]["position"][index],
                "limits": camera_limits[f"gripper_{axis}"],
            })
        camera_panel = SliderPanel(cv2, "Camera controls", controls, 288)

    if joint_gui:
        joint_limits = config["robot"]["joint_limits"] + [[config["robot"]["gripper_lower"], config["robot"]["gripper_upper"]]]
        controls = []
        for index, (value, limits) in enumerate(zip(action[0], joint_limits)):
            label = f"Joint {index + 1}" if index < 7 else "Gripper"
            controls.append({"label": label, "value": float(value), "limits": limits})
        joint_panel = SliderPanel(cv2, "Joint controls", controls)

    steps = simulation["steps"] if args.steps is None else args.steps
    step_iterator = range(steps) if steps > 0 else itertools.count()
    camera_values = None
    front_pose = None
    try:
        for _ in step_iterator:
            camera_positions = None
            if camera_panel is not None:
                values = camera_panel.values()
                camera_values = values
                front_pose = front_camera_from_controls(
                    front_initial_eye, front_initial_target, values[:5]
                )
                env.unwrapped.set_camera_positions(front_pose[0], values[5:], front_pose[1])

            if joint_panel is not None:
                action[0] = joint_panel.values()

            env.step(action)
            if not args.headless:
                env.unwrapped.show_camera_markers()
                env.unwrapped.render_human()

            if camera_panel is not None:
                frame = camera_frame(env, camera_names)[0].cpu().numpy()
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                cv2.imshow(camera_panel.window, camera_panel.draw(frame))

            if joint_panel is not None:
                cv2.imshow(joint_panel.window, joint_panel.draw())

            if cv2 is not None:
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("s") and args.task and camera_values is not None:
                    config["cameras"]["base_camera"]["eye"] = front_pose[0]
                    config["cameras"]["base_camera"]["target"] = front_pose[1]
                    config["cameras"]["wrist_camera"]["position"] = [float(value) for value in camera_values[5:]]
                    path = Path(__file__).resolve().parents[1] / "tasks" / args.task / "config.yaml"
                    path.write_text(yaml.safe_dump(config, sort_keys=False))
                    print(f"saved camera positions to {path}")
    except KeyboardInterrupt:
        print("Stopping simulation.")

    if args.headless:
        import imageio.v3 as iio

        output = Path(config["render"]["camera_output"])
        output.parent.mkdir(exist_ok=True)
        iio.imwrite(output, camera_frame(env)[0].cpu().numpy())
        print(f"Saved {output}")

    if cv2 is not None:
        cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    main()
