# Simulates picking up a loose light bulb and threading it into its HiveBoard socket.
# Created: 2026-10-22

from env.fr3_table_env import load_task_config
from mani_skill.utils.registration import register_env
from tasks.hiveboard_thread import ThreadFR3Env

CONFIG = load_task_config(__file__)


@register_env("LightBulbFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class LightBulbFR3Env(ThreadFR3Env):
    config = CONFIG
