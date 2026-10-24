# Prepares datasets for diffusion policy training.
# Created: 2026-09-14

import shutil
from pathlib import Path

import numpy as np
import torch
import zarr

from baselines.rollout import CONTROL_HZ, STATE_DIM
from env.fr3_table_env import PROJECT_ROOT
from tasks import TASKS
from tasks.generate_demos import RECORDING_NAME, resample_episodes, write_zarr


# Return the task's baseline zarr store, building it once from the raw (lossless) recording.
def ensure_dataset(task, record_dir, hz, cameras):
    demo_dir = PROJECT_ROOT / record_dir / TASKS[task]
    out_dir = demo_dir / f"baseline_zarr_{hz}hz"
    store = out_dir / "dataset.zarr"
    if (store / "meta").exists():
        return store
    recording = demo_dir / RECORDING_NAME
    if not recording.exists():
        raise FileNotFoundError(f"no recording at {recording}; run `make demos {task}` first")
    tmp_dir = out_dir.with_name(out_dir.name + "_tmp")
    shutil.rmtree(tmp_dir, ignore_errors=True)
    tmp_dir.mkdir(parents=True)
    config = dict(hz=hz, actions=["joint_position"], cameras=list(cameras))
    write_zarr(tmp_dir, resample_episodes(recording, CONTROL_HZ, config), hz)
    shutil.rmtree(out_dir, ignore_errors=True)
    tmp_dir.rename(out_dir)
    return store


# Separate recorded episodes into training and validation sets.
def split_episodes(store, num_episodes, val_episodes):
    total = zarr.open_group(Path(store), mode="r")["meta/episode_ends"].shape[0]
    total = min(total, num_episodes or total)
    if not 0 <= val_episodes < total:
        raise ValueError(f"val_episodes must be between 0 and {total - 1}")
    return list(range(total - val_episodes)), list(range(total - val_episodes, total))


class SequenceDataset(torch.utils.data.Dataset):
    """Windows of `horizon` steps padded at episode edges, as in DP's SequenceSampler, over the given episodes."""

    def __init__(self, store, cameras, horizon, n_obs_steps, n_action_steps, episodes):
        root = zarr.open_group(Path(store), mode="r")
        ends = root["meta/episode_ends"][:]
        starts = np.concatenate([[0], ends[:-1]])
        self.store = Path(store)
        self.cameras = list(cameras)
        self.n_obs_steps = n_obs_steps
        self.n_action_steps = n_action_steps
        self.num_episodes = len(episodes)
        n = int(ends[max(episodes)]) if episodes else 0
        self.state = root["data/qpos"][:n, :STATE_DIM].astype(np.float32)
        self.action = np.concatenate(
            [root["data/action_joint_position"][:n], root["data/action_gripper"][:n]], axis=1
        ).astype(np.float32)
        self.episode_bounds = [(int(starts[e]), int(ends[e])) for e in episodes]
        windows = []
        for start, end in self.episode_bounds:
            length = end - start
            for i in range(-(n_obs_steps - 1), length - horizon + n_action_steps):
                windows.append(start + np.clip(np.arange(i, i + horizon), 0, length - 1))
        self.windows = np.stack(windows) if windows else np.zeros((0, horizon), dtype=int)
        self.images = None

    def stats(self):
        frames = np.concatenate([np.arange(start, end) for start, end in self.episode_bounds])
        return dict(
            action_min=self.action[frames].min(0), action_max=self.action[frames].max(0),
            state_min=self.state[frames].min(0), state_max=self.state[frames].max(0),
        )

    def frames(self, camera, indices):
        if self.images is None:
            root = zarr.open_group(self.store, mode="r")
            self.images = {c: root[f"data/{c}"] for c in self.cameras}
        return torch.from_numpy(np.stack([self.images[camera][int(t)] for t in indices]))

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        window = self.windows[i]
        obs_steps = window[: self.n_obs_steps]
        return dict(
            images={camera: self.frames(camera, obs_steps) for camera in self.cameras},
            state=torch.from_numpy(self.state[obs_steps]),
            action=torch.from_numpy(self.action[window]),
        )
