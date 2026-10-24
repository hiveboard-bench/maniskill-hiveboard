# Simulates turning a ball valve whose friction ring increases the force needed at the lever.
# Created: 2026-10-22

from env.fr3_table_env import load_task_config
from mani_skill.utils.registration import register_env
from tasks.ball_valve.task import BallValveFR3Env

CONFIG = load_task_config(__file__)


@register_env("BallValveRingFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class BallValveRingFR3Env(BallValveFR3Env):
    config = CONFIG
