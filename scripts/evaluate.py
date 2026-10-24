# Evaluates policies across robot tasks.
# Created: 2026-09-18

import argparse
import json
import statistics
import time
from collections import Counter
from contextlib import ExitStack
from pathlib import Path

import torch
import yaml

from baselines.rollout import rollout, task_overrides
from env.fr3_table_env import PROJECT_ROOT

DEFAULTS = ("episodes", "seed_offset", "max_steps", "num_envs")
CAMERAS = ["base_camera", "wrist_camera"]


# Return (policy, cameras, hz) for a run; pi05 connections are closed by the exit stack.
def load_policy(run, device, stack):
    if run["policy"] == "diffusion_policy":
        from baselines.diffusion_policy.evaluate import load_policy as load_dp

        policy, config = load_dp(PROJECT_ROOT / run["checkpoint"], device)
        return policy, config["dataset"]["cameras"], config["dataset"]["hz"]
    if run["policy"] == "pi05":
        from baselines.pi05.evaluate import RemotePolicy, connect

        connection = stack.enter_context(connect(run.get("host", "localhost"), run.get("port", 8000)))
        return RemotePolicy(connection, run.get("action_steps", 10)), CAMERAS, run.get("hz", 20)
    raise ValueError(f"unknown policy {run['policy']!r}; use diffusion_policy or pi05")


# Run policy episodes for one task and collect outcomes.
def evaluate_task(policy, cameras, hz, task, settings, overrides, device):
    with task_overrides(task, overrides, settings["max_steps"]) as env_class:
        return rollout(
            policy, task, cameras, hz, settings["episodes"], settings["seed_offset"],
            0, device, settings["max_steps"], settings["num_envs"],
        ), env_class.failure_keys


# Aggregate success and failure counts across evaluation runs.
def summarize(metrics, records, failure_keys):
    failed = [r for r in records if not r["success"]]
    return {
        "episodes": len(records),
        "successes": len(records) - len(failed),
        "failures": len(failed),
        "outcomes": dict(Counter(r["outcome"] for r in records)),
        "failure_counts": {key: sum(r[key] for r in records) for key in failure_keys},
        **metrics,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config" / "eval.yaml")
    parser.add_argument("--runs", nargs="*", help="evaluate only these run names, even if disabled")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text())
    runs = [r for r in config["runs"] if (r["name"] in args.runs if args.runs else r.get("enabled", True))]
    if not runs:
        raise SystemExit("no runs selected; enable one in the config or pass --runs")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_path = PROJECT_ROOT / config["output_dir"] / f"results_{time.strftime('%Y%m%d_%H%M%S')}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = {"config": config, "results": [], "aggregate": []}

    for run in runs:
        settings = {key: run.get(key, config[key]) for key in DEFAULTS}
        overrides = run.get("task_overrides", {})
        seed_runs = [run | entry for entry in run.get("seeds", [{}])]
        rates = {task: [] for task in run["tasks"]}
        for index, seed_run in enumerate(seed_runs):
            with ExitStack() as stack:
                policy, cameras, hz = load_policy(seed_run, device, stack)
                for task in run["tasks"]:
                    print(f"{run['name']} seed {index} on {task}: {settings['episodes']} episodes from seed {settings['seed_offset']}")
                    (metrics, _, records), failure_keys = evaluate_task(
                        policy, cameras, hz, task, settings, overrides, device
                    )
                    summary = summarize(metrics, records, failure_keys)
                    rates[task].append(summary["success_rate"])
                    results["results"].append({
                        "run": run["name"],
                        "policy": run["policy"],
                        "seed_index": index,
                        "checkpoint": seed_run.get("checkpoint"),
                        "task": task,
                        **settings,
                        "task_overrides": overrides,
                        "summary": summary,
                        "episode_results": records,
                    })
                    out_path.write_text(json.dumps(results, indent=2))
                    print(f"  success {summary['successes']}/{summary['episodes']}, outcomes {summary['outcomes']}")
        for task, values in rates.items():
            std = statistics.stdev(values) if len(values) > 1 else None
            results["aggregate"].append({
                "run": run["name"], "policy": run["policy"], "task": task, "training_seeds": len(values),
                "success_rates": values, "success_rate_mean": statistics.mean(values), "success_rate_std": std,
            })
            out_path.write_text(json.dumps(results, indent=2))
            print(f"{run['name']} on {task}: {statistics.mean(values):.3f}" + (f" ± {std:.3f}" if std is not None else "") + f" over {len(values)} seeds")
    print(f"saved {out_path}")


if __name__ == "__main__":
    main()
