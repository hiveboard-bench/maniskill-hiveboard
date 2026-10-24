# Simulates picking up an M8 nut and threading it down onto a HiveBoard screw.
# Created: 2026-10-22

from env.fr3_table_env import load_task_config
from mani_skill.utils.registration import register_env
from tasks.hiveboard_thread import ThreadFR3Env

CONFIG = load_task_config(__file__)


@register_env("ThreadM8FR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class ThreadM8FR3Env(ThreadFR3Env):
    config = CONFIG
