# Serves a pi0.5 policy for inference.
# Created: 2026-09-16

import argparse
import logging

from openpi.policies import policy_config
from openpi.serving import websocket_policy_server

from baselines.pi05.config import CONFIGS, make_config


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("config", choices=CONFIGS)
    parser.add_argument("--checkpoint", required=True, help="step folder, e.g. outputs/baselines/pi05/checkpoints/pi05_fr3_lora/<exp>/29999")
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, force=True)

    config = make_config(args.config, args.repo_id, exp_name="serve", wandb=False)
    policy = policy_config.create_trained_policy(config, args.checkpoint)
    server = websocket_policy_server.WebsocketPolicyServer(policy=policy, host="0.0.0.0", port=args.port, metadata=policy.metadata)
    server.serve_forever()


if __name__ == "__main__":
    main()
