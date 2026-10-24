# Runs policy rollouts and records evaluation results.
# Created: 2026-09-16

import copy
import hashlib
import json
from collections import deque
from contextlib import contextmanager
from pathlib import Path

import gymnasium as gym
import h5py
import imageio.v2 as iio
import numpy as np
import torch
import torch.nn.functional as F

from env.fr3_table_env import CONFIG
from tasks import ENV_CLASSES, TASKS

CONTROL_HZ = 20
STATE_DIM = ACTION_DIM = 8


# Temporarily override a task's config; task configs are class attributes read at reset.
@contextmanager
def task_overrides(task, overrides, max_steps=None):
    env_class = ENV_CLASSES[task]
    original = env_class.config
    env_class.config = copy.deepcopy(original)
    try:
        for key, value in overrides.items():
            *parents, leaf = key.split(".")
            node = env_class.config
            for parent in parents:
                node = node[parent]
            if leaf not in node:
                raise KeyError(f"task override {key!r} does not match an existing config key")
            node[leaf] = value
        if max_steps:
            env_class.config["episode"]["max_episode_steps"] = max_steps
        yield env_class
    finally:
        env_class.config = original


# 95% confidence interval of a success rate, which stays meaningful for small numbers of episodes.
def wilson_interval(successes, trials, z=1.96):
    if trials == 0:
        return 0.0, 0.0
    p = successes / trials
    center = (p + z * z / (2 * trials)) / (1 + z * z / trials)
    half = z * np.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials)) / (1 + z * z / trials)
    return float(center - half), float(center + half)


# Create a ManiSkill environment with evaluation task overrides.
def make_env(task, num_envs, sim_backend, max_steps):
    kwargs = dict(max_episode_steps=max_steps) if max_steps else {}
    return gym.make(
        TASKS[task], num_envs=num_envs, obs_mode="rgb", control_mode="pd_joint_pos", sim_backend=sim_backend, **kwargs
    )


_EVALUATION_ENVS = {}


# Reuse one environment across rollout batches to avoid renderer failures.
def evaluation_env(task, num_envs, sim_backend, max_steps):
    key = (task, num_envs, sim_backend, max_steps)
    if key not in _EVALUATION_ENVS:
        _EVALUATION_ENVS[key] = make_env(task, num_envs, sim_backend, max_steps)
    env = _EVALUATION_ENVS[key]
    env.unwrapped.camera_history = {}
    return env


# Run the policy with pd_joint_pos on the GPU simulator and return (metrics, videos, records).
def rollout(
    policy, task, cameras, hz, episodes, seed_offset, video_episodes, device, max_steps=None, num_envs=16,
    policy_seed_offset=None, record_dir=None, record_metadata=None, sim_backend="gpu", denoise_variance=None,
    return_recorded=False, episode_seeds=None, snapshot_every=None, resume_snapshot=None,
    resample_policy_seed=None, reuse_env=False,
):
    stride = CONTROL_HZ // hz
    latency = int(round(CONFIG["latency"]["inference_seconds"] * CONTROL_HZ))
    num_envs = 1 if sim_backend == "cpu" else min(num_envs, episodes)
    if (snapshot_every or resume_snapshot) and (episodes != 1 or num_envs != 1):
        raise ValueError("snapshot capture and resume require one episode and one environment")
    if snapshot_every and record_dir is None:
        raise ValueError("snapshot capture requires record_dir")
    if episode_seeds is not None and len(episode_seeds) != episodes:
        raise ValueError("episode_seeds must contain one reset seed per episode")
    env = evaluation_env(task, num_envs, sim_backend, max_steps) if reuse_env else make_env(task, num_envs, sim_backend, max_steps)
    failure_keys = env.unwrapped.failure_keys
    results, videos, recorded = [], [], []
    for start in range(0, episodes, num_envs):
        seeds = (
            [int(episode_seeds[i]) for i in range(start, min(start + num_envs, episodes))]
            if episode_seeds is not None
            else [seed_offset + i for i in range(start, min(start + num_envs, episodes))]
        )
        obs, _ = env.reset(seed=seeds + [seeds[-1]] * (num_envs - len(seeds)))
        if policy_seed_offset is not None:
            torch.manual_seed(policy_seed_offset + start)
        history = deque(maxlen=(policy.n_obs_steps - 1) * stride + 1)
        actions = deque()
        action_sources = deque()
        pending, arrival, last = None, 0, None
        last_source_ids, last_chunk_index = [None] * len(seeds), -1
        finished = {}
        step_counts = [0] * len(seeds)
        episode_data = ([dict(states=[], velocities=[], joint_torques=[], executed_actions=[], decision_ids=[], chunk_indices=[], decisions=[],
                              executed_variance=[], first_failure_timesteps={})
                         for _ in seeds] if record_dir is not None or return_recorded else None)
        if resume_snapshot is not None:
            snapshot = torch.load(resume_snapshot, weights_only=False)
            if snapshot["seed"] != seeds[0]:
                raise ValueError("resume seed must match the snapshot reset seed")
            env.unwrapped.set_state_dict(snapshot["environment"])
            env.unwrapped._elapsed_steps = snapshot["elapsed_steps"]
            if hasattr(env, "_elapsed_steps"):
                env._elapsed_steps = snapshot["wrapper_elapsed_steps"]
            if snapshot.get("camera_history") is not None:
                env.unwrapped.camera_history = snapshot["camera_history"]
            obs = snapshot["observation"]
            history = deque(snapshot["history"], maxlen=history.maxlen)
            actions = deque(snapshot["actions"])
            action_sources = deque([([None], -1)] * len(actions))
            pending = snapshot["pending"]
            if pending is not None:
                pending_sources = [([None], -1)] * len(pending)
            arrival = snapshot["arrival"]
            last = snapshot["last"]
            step_counts[0] = snapshot["step_count"]
            torch.set_rng_state(snapshot["torch_rng_cpu"].cpu())
            if torch.cuda.is_available() and snapshot.get("torch_rng_cuda") is not None:
                torch.cuda.set_rng_state(snapshot["torch_rng_cuda"].cpu())
            if resample_policy_seed is not None:
                torch.manual_seed(int(resample_policy_seed))
            if episode_data is not None:
                episode_data[0]["initial_qpos"] = snapshot["initial_qpos"]
                episode_data[0]["initial_observation_sha256"] = snapshot["initial_observation_sha256"]
                episode_data[0]["initial_observation_signature"] = snapshot["initial_observation_signature"]
                episode_data[0]["resume_step"] = snapshot["step_count"]
        if episode_data is not None and resume_snapshot is None:
            for j, data in enumerate(episode_data):
                data["initial_qpos"] = obs["agent"]["qpos"][j, :STATE_DIM].cpu().numpy().tolist()
                data["initial_observation_sha256"] = {
                    camera: hashlib.sha256(
                        obs["sensor_data"][camera]["rgb"][j].cpu().numpy().tobytes()
                    ).hexdigest()
                    for camera in cameras
                }
                data["initial_observation_signature"] = {}
                for camera in cameras:
                    image = obs["sensor_data"][camera]["rgb"][j].permute(2, 0, 1).unsqueeze(0).float()
                    signature = F.adaptive_avg_pool2d(image, (16, 16))[0].permute(1, 2, 0)
                    data["initial_observation_signature"][camera] = (signature * 10).round().div(10).cpu().numpy().tolist()
        filmed = [j for j in range(len(seeds)) if start + j < video_episodes]
        frames = {j: [] for j in filmed} if record_dir is None else None
        video_writers = {}
        if record_dir is not None:
            video_dir = Path(record_dir) / "videos"
            video_dir.mkdir(parents=True, exist_ok=True)
            for j in filmed:
                episode_index = start + j
                video_writers[j] = iio.get_writer(video_dir / f"episode_{episode_index}.mp4", fps=CONTROL_HZ, codec="libx264")
        while len(finished) < len(seeds):
            history.append(obs)
            if pending is None and len(actions) <= latency:
                past = [history[max(0, len(history) - 1 - k * stride)] for k in reversed(range(policy.n_obs_steps))]
                images = {camera: torch.stack([o["sensor_data"][camera]["rgb"] for o in past], dim=1).to(device) for camera in cameras}
                state = torch.stack([o["agent"]["qpos"][:, :STATE_DIM] for o in past], dim=1).float().to(device)
                condition = None
                if denoise_variance is not None:
                    predicted, query_variance = policy.predict_action_with_variance(
                        images, state, env.unwrapped.prompts,
                        variance_steps=denoise_variance["last_steps"],
                        ddim_eta=denoise_variance.get("ddim_eta", 0.0),
                    )
                    condition = None
                elif record_dir is not None and hasattr(policy, "predict_action_with_features"):
                    predicted, condition = policy.predict_action_with_features(images, state, env.unwrapped.prompts)
                    query_variance = None
                else:
                    predicted = policy.predict_action(images, state, env.unwrapped.prompts)
                    query_variance = None
                pending = [predicted[:, k] for k in range(predicted.shape[1]) for _ in range(stride)]
                if episode_data is not None:
                    decision_ids = [None] * len(seeds)
                    for j in range(len(seeds)):
                        if j not in finished:
                            decision_ids[j] = len(episode_data[j]["decisions"])
                            episode_data[j]["decisions"].append({
                                "issue_step": step_counts[j],
                                "state": state[j].cpu().numpy(),
                                "predicted_action": predicted[j].cpu().numpy(),
                                "condition": condition[j].cpu().numpy() if condition is not None else None,
                                "denoising_variance": query_variance[j].cpu().numpy() if query_variance is not None else None,
                                "executed_index_start": policy.n_obs_steps - 1,
                                "executed_index_end": policy.n_obs_steps - 1 + policy.n_action_steps,
                            })
                    pending_sources = [(decision_ids, k) for k in range(predicted.shape[1]) for _ in range(stride)]
                arrival = latency
            if pending is not None and arrival == 0:
                actions = deque(pending[latency:] or pending[-1:])
                if episode_data is not None:
                    action_sources = deque(pending_sources[latency:] or pending_sources[-1:])
                pending = None
            arrival -= 1
            if actions:
                last = actions.popleft()
                if episode_data is not None:
                    source_ids, chunk_index = action_sources.popleft()
                    last_source_ids, last_chunk_index = source_ids, chunk_index
            elif last is None:
                last = obs["agent"]["qpos"][:, :ACTION_DIM].clone()
                if episode_data is not None:
                    source_ids, chunk_index = [None] * len(seeds), -1
            else:
                if episode_data is not None:
                    source_ids, chunk_index = last_source_ids, last_chunk_index
            if episode_data is not None:
                for j in range(len(seeds)):
                    if j in finished:
                        continue
                    episode_data[j]["states"].append(obs["agent"]["qpos"][j, :STATE_DIM].cpu().numpy())
                    episode_data[j]["velocities"].append(obs["agent"]["qvel"][j].cpu().numpy())
                    episode_data[j]["joint_torques"].append(env.unwrapped.agent.robot.get_qf()[j].cpu().numpy())
                    episode_data[j]["executed_actions"].append(last[j].cpu().numpy())
                    episode_data[j]["decision_ids"].append(-1 if source_ids[j] is None else source_ids[j])
                    episode_data[j]["chunk_indices"].append(chunk_index)
                    if denoise_variance is not None and source_ids[j] is not None:
                        full_index = policy.n_obs_steps - 1 + chunk_index
                        episode_data[j]["executed_variance"].append(
                            float(episode_data[j]["decisions"][source_ids[j]]["denoising_variance"][full_index])
                        )
                    else:
                        episode_data[j]["executed_variance"].append(float("nan"))
                    step_counts[j] += 1
            obs, _, terminated, truncated, info = env.step(last)
            done = (terminated | truncated).cpu()
            if episode_data is not None:
                for j in range(len(seeds)):
                    if j in finished:
                        continue
                    timestep = step_counts[j] - 1
                    for key in failure_keys:
                        if key not in episode_data[j]["first_failure_timesteps"] and key in info and bool(info[key][j]):
                            episode_data[j]["first_failure_timesteps"][key] = timestep
            for j in filmed:
                if j not in finished:
                    frame = torch.cat([obs["sensor_data"][camera]["rgb"][j] for camera in cameras], dim=1).cpu().numpy()
                    if record_dir is None:
                        frames[j].append(frame)
                    else:
                        video_writers[j].append_data(frame)
            if snapshot_every and not bool(done[0]) and step_counts[0] % int(snapshot_every) == 0:
                snapshot_dir = Path(record_dir) / "snapshots"
                snapshot_dir.mkdir(parents=True, exist_ok=True)
                data = episode_data[0]
                torch.save({
                    "seed": seeds[0], "step_count": step_counts[0],
                    "environment": env.unwrapped.get_state_dict(),
                    "elapsed_steps": env.unwrapped._elapsed_steps,
                    "wrapper_elapsed_steps": getattr(env, "_elapsed_steps", None),
                    "camera_history": getattr(env.unwrapped, "camera_history", None),
                    "observation": obs, "history": list(history), "actions": list(actions),
                    "pending": pending, "arrival": arrival, "last": last,
                    "torch_rng_cpu": torch.get_rng_state(),
                    "torch_rng_cuda": torch.cuda.get_rng_state() if torch.cuda.is_available() else None,
                    "initial_qpos": data["initial_qpos"],
                    "initial_observation_sha256": data["initial_observation_sha256"],
                    "initial_observation_signature": data["initial_observation_signature"],
                }, snapshot_dir / f"step_{step_counts[0]:06d}.pt")
            for j in range(len(seeds)):
                if done[j] and j not in finished:
                    finished[j] = episode_record(info, j, failure_keys) | {"seed": seeds[j]}
                    if j in video_writers:
                        video_writers.pop(j).close()
        results.extend(finished[j] for j in range(len(seeds)))
        if episode_data is not None:
            recorded.extend(episode_data[j] for j in range(len(seeds)))
        if record_dir is None:
            videos.extend(np.stack(frames[j]) for j in filmed)
    if not reuse_env:
        env.close()
    successes = [r for r in results if r["success"]]
    low, high = wilson_interval(len(successes), len(results))
    metrics = {
        "success_rate": len(successes) / len(results),
        "success_rate_ci_low": low,
        "success_rate_ci_high": high,
        "mean_steps": float(np.mean([r["steps"] for r in results])),
        "mean_steps_to_success": float(np.mean([r["steps"] for r in successes])) if successes else float("nan"),
    }
    metrics |= {f"{key}_rate": float(np.mean([r[key] for r in results])) for key in failure_keys}
    metrics |= {key: float(np.mean([r[key] for r in results])) for key in results[0] if key.startswith("final_")}
    if record_dir is not None:
        write_rollout_recording(record_dir, recorded, results, cameras, hz, video_episodes, record_metadata or {},
                                denoise_variance=denoise_variance)
    return (metrics, videos, results, recorded) if return_recorded else (metrics, videos, results)


# Write passive, aligned rollout arrays and episode labels to new sidecar files.
def write_rollout_recording(record_dir, episodes, records, cameras, hz, video_episodes, metadata=None,
                            denoise_variance=None):
    record_dir = Path(record_dir)
    record_dir.mkdir(parents=True, exist_ok=True)
    h5_path = record_dir / "trajectories.h5"
    manifest_path = record_dir / "manifest.json"
    if h5_path.exists() or manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite an existing recording under {record_dir}")
    with h5py.File(h5_path, "w") as output:
        for index, (episode, record) in enumerate(zip(episodes, records)):
            group = output.create_group(f"traj_{index}")
            observation_group = group.create_group("obs")
            action_group = group.create_group("actions")
            meta_group = group.create_group("meta")
            decision_group = group.create_group("decisions")
            observation_group.create_dataset("qpos", data=np.asarray(episode["states"], dtype=np.float32))
            observation_group.create_dataset("qvel", data=np.asarray(episode["velocities"], dtype=np.float32))
            observation_group.create_dataset("joint_torque", data=np.asarray(episode["joint_torques"], dtype=np.float32))
            action_group.create_dataset("executed", data=np.asarray(episode["executed_actions"], dtype=np.float32))
            meta_group.create_dataset("decision_id", data=np.asarray(episode["decision_ids"], dtype=np.int64))
            meta_group.create_dataset("action_chunk_index", data=np.asarray(episode["chunk_indices"], dtype=np.int64))
            if denoise_variance is not None:
                meta_group.create_dataset("executed_action_variance", data=np.asarray(episode["executed_variance"], dtype=np.float32))
            decisions = episode["decisions"]
            if decisions:
                decision_group.create_dataset("issue_step", data=np.asarray([d["issue_step"] for d in decisions], dtype=np.int64))
                decision_group.create_dataset("state", data=np.stack([d["state"] for d in decisions]).astype(np.float32))
                decision_group.create_dataset("predicted_action", data=np.stack([d["predicted_action"] for d in decisions]).astype(np.float32))
                if all(d["condition"] is not None for d in decisions):
                    decision_group.create_dataset("condition", data=np.stack([d["condition"] for d in decisions]).astype(np.float32))
                if denoise_variance is not None:
                    variances = np.stack([d["denoising_variance"] for d in decisions]).astype(np.float32)
                    decision_group.create_dataset("denoising_variance", data=variances)
                    start = int(episodes[index]["decisions"][0]["executed_index_start"])
                    end = int(episodes[index]["decisions"][0]["executed_index_end"])
                    actual_mean = np.full(len(decisions), np.nan, dtype=np.float32)
                    actual_max = np.full(len(decisions), np.nan, dtype=np.float32)
                    for decision_index in range(len(decisions)):
                        indices = sorted({chunk_index for source_id, chunk_index in zip(
                            episodes[index]["decision_ids"], episodes[index]["chunk_indices"]
                        ) if source_id == decision_index})
                        if indices:
                            executed = variances[decision_index, [start + k for k in indices]]
                            actual_mean[decision_index] = executed.mean()
                            actual_max[decision_index] = executed.max()
                    decision_group.create_dataset("executed_variance_mean", data=actual_mean)
                    decision_group.create_dataset("executed_variance_max", data=actual_max)
                    decision_group.create_dataset("executed_index_range", data=np.tile([start, end], (len(decisions), 1)))
            group.attrs["steps"] = len(episode["states"])

    episode_metadata = []
    for index, (record, episode) in enumerate(zip(records, episodes)):
        episode_metadata.append({
            "episode_id": f"traj_{index}",
            "success": bool(record["success"]),
            "outcome": record["outcome"],
            "failure_reasons": list(record["failure_reasons"]),
            "seed": int(record["seed"]),
            "steps": int(record["steps"]),
            "decision_count": len(episode["decisions"]),
            "initial_qpos": episode.get("initial_qpos"),
            "initial_observation_sha256": episode.get("initial_observation_sha256"),
            "initial_observation_signature": episode.get("initial_observation_signature"),
            "video": f"videos/episode_{index}.mp4" if index < video_episodes else None,
            "video_frame_count": int(record["steps"]) if index < video_episodes else None,
            "failure_subconditions": {key: bool(record[key]) for key in record if key not in {
                "success", "seed", "steps", "decision_count", "video", "video_frame_count", "outcome", "failure_reasons"
            } and not key.startswith("final_")},
            "first_failure_timesteps": episode.get("first_failure_timesteps", {}),
        })
    manifest = {
        "schema_version": 1,
        "recording_type": "policy_closed_loop",
        "read_only_source": True,
        "policy_hz": int(hz),
        "control_hz": CONTROL_HZ,
        "sim_backend": metadata.get("sim_backend", "gpu"),
        "action_stride_control_steps": CONTROL_HZ // hz,
        "inference_latency_steps": int(round(CONFIG["latency"]["inference_seconds"] * CONTROL_HZ)),
        "cameras": list(cameras),
        "video_fps": CONTROL_HZ,
        "video_frame_timing": "frame t is sampled from the post-action observation after executed action t; the camera image itself may be delayed",
        "camera_latency_steps": {
            camera: int(round(CONFIG["latency"]["camera_seconds"].get(camera, 0.0) * CONTROL_HZ))
            for camera in cameras
        },
        "video_layout": "camera views are concatenated left-to-right in the cameras list",
        "state_timing": "qpos[t] is recorded before executed action[t]",
        "force_timing": "qvel[t] and joint_torque[t] are read before executed action[t]; joint_torque is simulator generalized joint force, not an external wrench",
        "action_timing": "actions/executed is the action passed to env.step, including repeated chunk actions and initial holds",
        "decision_mapping": "meta/decision_id and meta/action_chunk_index map each executed step to the policy inference and predicted action index; -1 means no policy chunk source",
        "failure_timestep_convention": "first true sub-condition timestep is zero-based and aligned with the executed action row at that environment step",
        "metadata": metadata or {},
        "denoise_variance": metadata.get("denoise_variance") if metadata else None,
        "episodes": episode_metadata,
    }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    if denoise_variance is not None:
        data_dir = record_dir / "data"
        data_dir.mkdir(exist_ok=True)
        for index, (episode, record) in enumerate(zip(episodes, records)):
            decisions = []
            for decision_index, decision in enumerate(episode["decisions"]):
                values = decision["denoising_variance"]
                start, end = decision["executed_index_start"], decision["executed_index_end"]
                executed_pairs = [
                    (step, chunk_index)
                    for step, (source_id, chunk_index) in enumerate(zip(episode["decision_ids"], episode["chunk_indices"]))
                    if source_id == decision_index
                ]
                executed_indices = sorted({chunk_index for _, chunk_index in executed_pairs})
                executed_values = values[[start + k for k in executed_indices]] if executed_indices else np.asarray([])
                decisions.append({
                    "query_env_timestep": int(decision["issue_step"]),
                    "variance_by_chunk_index": values.tolist(),
                    "executed_index_range": [int(start), int(end)],
                    "actually_executed_chunk_indices": executed_indices,
                    "actually_executed_env_timesteps": [int(step) for step, _ in executed_pairs],
                    "executed_variance_mean": float(np.mean(executed_values)) if len(executed_values) else None,
                    "executed_variance_max": float(np.max(executed_values)) if len(executed_values) else None,
                })
            payload = {
                "episode_id": f"traj_{index}", "seed": int(record["seed"]), "steps": int(record["steps"]),
                "outcome": record["outcome"], "task": (metadata or {}).get("task"),
                "checkpoint": (metadata or {}).get("checkpoint"), "scheduler": (metadata or {}).get("denoise_variance", {}),
                "inference_steps": (metadata or {}).get("denoise_variance", {}).get("inference_steps"),
                "last_L": denoise_variance["last_steps"], "success": bool(record["success"]),
                "failure_subconditions": {key: bool(record[key]) for key in record if key not in {
                    "success", "seed", "steps", "outcome", "failure_reasons"
                } and not key.startswith("final_")},
                "first_failure_timesteps": episode.get("first_failure_timesteps", {}),
                "executed_variance_by_env_timestep": [
                    None if not np.isfinite(value) else float(value) for value in episode["executed_variance"]
                ],
                "queries": decisions,
            }
            (data_dir / f"ep_{index}.json").write_text(json.dumps(payload, indent=2) + "\n")


# Outcome of env j from the info of the step it finished.
def episode_record(info, j, failure_keys):
    record = {key: bool(info[key][j]) for key in ("success", *failure_keys)}
    causes = [key for key in failure_keys if record[key] and key != "timeout"]
    record["failure_reasons"] = causes
    record["outcome"] = "success" if record["success"] else (causes[0] if causes else "timeout" if record.get("timeout") else "failure")
    record["steps"] = int(info["elapsed_steps"][j])
    record |= {f"final_{key}": float(value[j]) for key, value in info.items()
               if key != "elapsed_steps" and torch.is_tensor(value) and value.dtype != torch.bool and value.dim() == 1}
    return record
