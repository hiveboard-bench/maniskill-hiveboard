# Evaluates pi0.5 policies.
# Created: 2026-09-16

import argparse
import functools
import json
import time
from pathlib import Path

import imageio.v3 as iio
import msgpack
import numpy as np
import torch
import websockets.sync.client

from baselines.rollout import CONTROL_HZ, rollout
from env.fr3_table_env import PROJECT_ROOT


# Encode a NumPy array for the policy server protocol.
def pack_array(obj):
    if isinstance(obj, np.ndarray):
        return {b"__ndarray__": True, b"data": obj.tobytes(), b"dtype": obj.dtype.str, b"shape": obj.shape}
    if isinstance(obj, np.generic):
        return {b"__npgeneric__": True, b"data": obj.item(), b"dtype": obj.dtype.str}
    return obj


# Decode an array returned by the policy server.
def unpack_array(obj):
    if b"__ndarray__" in obj:
        return np.ndarray(buffer=obj[b"data"], dtype=np.dtype(obj[b"dtype"]), shape=obj[b"shape"])
    if b"__npgeneric__" in obj:
        return np.dtype(obj[b"dtype"]).type(obj[b"data"])
    return obj


packb = functools.partial(msgpack.packb, default=pack_array)
unpackb = functools.partial(msgpack.unpackb, object_hook=unpack_array)


# Wait up to 10 minutes for the policy server, which can take a while to load a checkpoint.
def connect(host, port):
    for _ in range(120):
        try:
            return websockets.sync.client.connect(f"ws://{host}:{port}", compression=None, max_size=None)
        except OSError:
            time.sleep(5)
    raise ConnectionError(f"no policy server at ws://{host}:{port}")


class RemotePolicy:
    """pi0.5 queried over openpi's websocket protocol; executes the first `action_steps` actions of each chunk."""

    n_obs_steps = 1

    def __init__(self, connection, action_steps):
        self.connection = connection
        self.action_steps = action_steps
        self.metadata = unpackb(connection.recv())

    # The server takes one observation per request, so the batch is queried env by env.
    def predict_action(self, images, state, prompts):
        base = images["base_camera"][:, -1].cpu().numpy()
        wrist = images["wrist_camera"][:, -1].cpu().numpy()
        state = state[:, -1].cpu().numpy()
        chunks = []
        for i in range(len(state)):
            obs = {"observation/image": base[i], "observation/wrist_image": wrist[i], "observation/state": state[i], "prompt": prompts[i]}
            self.connection.send(packb(obs))
            response = self.connection.recv()
            if isinstance(response, str):
                raise RuntimeError(f"policy server error:\n{response}")
            chunks.append(np.array(unpackb(response)["actions"][: self.action_steps], dtype=np.float32))
        return torch.from_numpy(np.stack(chunks))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", required=True)
    parser.add_argument("--host", default="localhost")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--action-steps", type=int, default=10, help="actions executed from each predicted chunk before replanning")
    parser.add_argument("--hz", type=int, default=20, help="rate of the export the policy was trained on")
    parser.add_argument("--seed-offset", type=int, default=100000)
    parser.add_argument("--videos", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=16, help="episodes simulated in parallel on the GPU")
    parser.add_argument("--record-dir", type=Path, help="write aligned rollout arrays, labels, and selected videos to a new sidecar directory")
    args = parser.parse_args()
    if args.record_dir and args.record_dir.exists() and any(args.record_dir.iterdir()):
        parser.error(f"record directory must be empty: {args.record_dir}")

    with connect(args.host, args.port) as connection:
        policy = RemotePolicy(connection, args.action_steps)
        metrics, videos, _ = rollout(
            policy, args.task, ["base_camera", "wrist_camera"], args.hz, args.episodes, args.seed_offset,
            args.videos, "cpu", args.max_steps, args.num_envs, record_dir=args.record_dir,
            record_metadata={
                "policy_family": "pi05",
                "task": args.task,
                "server": {"host": args.host, "port": args.port},
                "action_steps": args.action_steps,
                "hz": args.hz,
                "seed_offset": args.seed_offset,
                "episodes": args.episodes,
                "num_envs": args.num_envs,
            },
        )
    out_dir = args.record_dir or PROJECT_ROOT / "outputs" / "baselines" / "pi05" / "eval" / f"{args.task}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir.mkdir(parents=True)
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    for i, video in enumerate(videos):
        iio.imwrite(out_dir / f"episode_{i}.mp4", video, fps=CONTROL_HZ, codec="libx264")
    print(json.dumps(metrics, indent=2))
    print(f"saved {out_dir}")


if __name__ == "__main__":
    main()
