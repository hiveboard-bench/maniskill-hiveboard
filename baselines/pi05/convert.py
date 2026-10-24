# Converts demonstrations for pi0.5 training.
# Created: 2026-09-16

import argparse
import shutil
from pathlib import Path

import h5py
import imageio.v3 as iio
import numpy as np
from lerobot.common.datasets.lerobot_dataset import HF_LEROBOT_HOME, LeRobotDataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("export_dir", type=Path, help="export_hdf5_<hz>hz folder written by tasks.generate_demos")
    parser.add_argument("--repo-id", required=True, help="LeRobot dataset id, e.g. your_hf_username/fr3_key_lock")
    parser.add_argument("--max-episodes", type=int, default=None)
    parser.add_argument("--push-to-hub", action="store_true")
    args = parser.parse_args()

    output_path = HF_LEROBOT_HOME / args.repo_id
    if output_path.exists():
        shutil.rmtree(output_path)

    with h5py.File(args.export_dir / "episodes.h5", "r") as f:
        first = f[next(iter(f))]
        if "action_joint_position" not in first:
            raise KeyError("export.actions in config/data.yaml must include joint_position")
        height, width = iio.imread(args.export_dir / first.attrs["base_camera_video"], index=0).shape[:2]
        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            robot_type="fr3",
            fps=int(f.attrs["hz"]),
            features={
                "image": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
                "wrist_image": {"dtype": "image", "shape": (height, width, 3), "names": ["height", "width", "channel"]},
                "state": {"dtype": "float32", "shape": (8,), "names": ["state"]},
                "actions": {"dtype": "float32", "shape": (8,), "names": ["actions"]},
            },
            image_writer_threads=10,
            image_writer_processes=5,
        )
        for name in sorted(f, key=lambda n: int(n.split("_")[1]))[: args.max_episodes]:
            episode = f[name]
            base = iio.imread(args.export_dir / episode.attrs["base_camera_video"])
            wrist = iio.imread(args.export_dir / episode.attrs["wrist_camera_video"])
            state = episode["qpos"][:, :8].astype(np.float32)
            actions = np.concatenate([episode["action_joint_position"], episode["action_gripper"]], axis=1).astype(np.float32)
            if not len(base) == len(wrist) == len(state):
                raise ValueError(f"{name}: {len(base)} base frames, {len(wrist)} wrist frames, {len(state)} steps")
            for k in range(len(state)):
                dataset.add_frame(
                    {
                        "image": base[k],
                        "wrist_image": wrist[k],
                        "state": state[k],
                        "actions": actions[k],
                        "task": str(episode.attrs["prompt"]),
                    }
                )
            dataset.save_episode()
            print(f"{name}: {len(state)} frames, prompt={episode.attrs['prompt']!r}")

    print(f"wrote {output_path}")
    if args.push_to_hub:
        dataset.push_to_hub(tags=["fr3", "maniskill"], private=True, push_videos=True)


if __name__ == "__main__":
    main()
