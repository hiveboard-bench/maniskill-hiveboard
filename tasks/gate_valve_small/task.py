# Simulates grasping the rim of a small gate valve wheel and turning it through a full revolution.
# Created: 2026-10-16

from env.fr3_table_env import load_task_config
from mani_skill.utils.registration import register_env
from tasks.hiveboard import HiveboardFR3Env

CONFIG = load_task_config(__file__)


@register_env("GateValveSmallFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class GateValveSmallFR3Env(HiveboardFR3Env):
    """Grasp the small handwheel across its rim and turn it a full revolution, regrasping between segments."""

    config = CONFIG
    failure_keys = ("never_contacted", "stopped_short", "unwound", "timeout")
