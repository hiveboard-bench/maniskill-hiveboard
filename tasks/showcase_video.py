# Composes exterior, front, and wrist camera frames into per-episode showcase videos.
# Created: 2026-10-22

from pathlib import Path

import imageio
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont, ImageOps

from env.fr3_table_env import CONFIG, REFLEX_KEYS

GAP = 6
BACKGROUND = (17, 18, 20)
BADGES = {"success": (46, 160, 67), "timeout": (214, 150, 20), "failed": (200, 55, 55)}
HOLD_SECONDS = 1.5


# Convert a tensor or image input to a NumPy array.
def _numpy(image) -> np.ndarray:
    return image.detach().cpu().numpy() if torch.is_tensor(image) else np.asarray(image)


class ShowcaseVideo:
    """Collects one composed frame per control step for each env and writes them as an MP4 per episode."""

    # Set up the state and resources used by this instance.
    def __init__(self, env, task: str, out_dir: Path):
        self.base = env.unwrapped
        if not getattr(self.base, "showcase_camera", False):
            raise ValueError("make the env with showcase_camera=True to record showcase videos")
        cfg = CONFIG["showcase_video"]
        self.cfg = cfg
        self.task = task.replace("_", " ")
        self.out_dir = Path(out_dir)
        self.cell = cfg["cell"]
        self.big = 2 * self.cell + GAP
        content = (3 * self.cell + 4 * GAP, 2 * self.cell + 3 * GAP)
        self.size = tuple(-(-side // 16) * 16 for side in content)
        self.origin = tuple((size - side) // 2 for size, side in zip(self.size, content))
        self.fps = self.base.control_freq
        self.limit = self.base.config["episode"]["max_episode_steps"] / self.fps
        self.title_font = ImageFont.load_default(size=26)
        self.small_font = ImageFont.load_default(size=17)
        self.badge_font = ImageFont.load_default(size=44)
        self.frames = {}
        self.tcp = {}
        self.step = 0

    # Start a new recording from the initial observation.
    def start(self, obs, rows):
        self.step = 0
        for row in rows:
            self.frames[row] = []
            self.tcp[row] = []
        self.capture(obs, rows)

    # Collect the current camera frames for each active episode.
    def capture(self, obs, rows):
        camera = self.base._human_render_cameras["showcase_camera"].camera
        camera.set_local_pose(self.base.showcase_pose(self.step / self.fps).sp)
        self.base.scene.update_render(update_sensors=False, update_human_render_cameras=True)
        showcase = _numpy(self.base.scene.get_human_render_camera_images("showcase_camera")["showcase_camera"])
        front = _numpy(obs["sensor_data"]["base_camera"]["rgb"])
        wrist = _numpy(obs["sensor_data"]["wrist_camera"]["rgb"])
        tcp = _numpy(self.base.agent.tcp.pose.p)
        for row in rows:
            self.frames[row].append(self._compose(showcase[row], front[row], wrist[row], self.step / self.fps))
            self.tcp[row].append(tcp[row])
        self.step += 1

    # Arrange the exterior, front, and wrist images into one frame.
    def _compose(self, showcase, front, wrist, seconds):
        canvas = Image.new("RGB", self.size, BACKGROUND)
        x, y = self.origin
        small = (self.cell, self.cell)
        fit = lambda image: ImageOps.pad(Image.fromarray(image[..., :3]), small, Image.Resampling.LANCZOS, color=BACKGROUND)
        canvas.paste(fit(wrist), (x + GAP, y + GAP))
        canvas.paste(fit(front), (x + GAP, y + 2 * GAP + self.cell))
        big = Image.fromarray(showcase[..., :3]).resize((self.big, self.big), Image.Resampling.LANCZOS)
        canvas.paste(big, (x + 2 * GAP + self.cell, y + GAP))
        draw = ImageDraw.Draw(canvas, "RGBA")
        self._label(draw, (x + GAP + 8, y + GAP + 8), "wrist", self.small_font)
        self._label(draw, (x + GAP + 8, y + 2 * GAP + self.cell + 8), "front", self.small_font)
        left = x + 2 * GAP + self.cell + 12
        self._label(draw, (left, y + GAP + 12), self.task, self.title_font)
        self._label(draw, (left, y + GAP + 52), f"{seconds:5.1f} s / {self.limit:.0f} s", self.small_font)
        return canvas

    # Draw task and outcome labels on a video frame.
    @staticmethod
    def _label(draw, xy, text, font, fill=(0, 0, 0, 120)):
        x0, y0, x1, y1 = draw.textbbox(xy, text, font=font)
        draw.rounded_rectangle((x0 - 7, y0 - 5, x1 + 7, y1 + 5), radius=6, fill=fill)
        draw.text(xy, text, font=font, fill=(255, 255, 255))

    # Whether the TCP moved less than stuck_meters over the episode's last stuck_seconds.
    def _stuck(self, row) -> bool:
        tail = np.asarray(self.tcp[row][-int(self.cfg["stuck_seconds"] * self.fps):])
        return len(tail) > 1 and np.linalg.norm(tail.max(axis=0) - tail.min(axis=0)) < self.cfg["stuck_meters"]

    # Write each completed episode as a labeled MP4 video.
    def save(self, row, seed, info) -> Path | None:
        flag = lambda key: key in info and bool(info[key].reshape(-1)[0])
        status = "success" if flag("success") else "timeout" if flag("timeout") else "failed"
        frames = self.frames.pop(row)
        stuck = status == "timeout" and self._stuck(row)
        self.tcp.pop(row)
        if stuck:
            return None
        reflexes = [key.removeprefix("reflex_") for key in REFLEX_KEYS if flag(key)]
        last = frames[-1].copy()
        draw = ImageDraw.Draw(last, "RGBA")
        center = (self.origin[0] + 2 * GAP + self.cell + self.big // 2, self.origin[1] + GAP + self.big // 2)
        text = status.upper()
        x0, y0, x1, y1 = draw.textbbox(center, text, font=self.badge_font, anchor="mm")
        draw.rounded_rectangle((x0 - 24, y0 - 16, x1 + 24, y1 + 16), radius=14, fill=(*BADGES[status], 225))
        draw.text(center, text, font=self.badge_font, fill=(255, 255, 255), anchor="mm")
        if reflexes:
            self._label(draw, (center[0] - 90, y1 + 34), "reflex: " + ", ".join(reflexes), self.small_font,
                        fill=(200, 55, 55, 200))
        frames += [last] * int(HOLD_SECONDS * self.fps)
        path = self.out_dir / f"seed{seed}_{status}{'_reflex' if reflexes else ''}.mp4"
        path.parent.mkdir(parents=True, exist_ok=True)
        with imageio.get_writer(path, fps=self.fps, codec="libx264", quality=8) as writer:
            for frame in frames:
                writer.append_data(np.asarray(frame))
        return path
