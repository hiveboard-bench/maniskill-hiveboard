# Simulates turning a large gate valve wheel through a full revolution with repeated grasps.
# Created: 2026-10-16

from env.fr3_table_env import load_task_config
from mani_skill.utils.registration import register_env
from tasks.hiveboard import HiveboardFR3Env

CONFIG = load_task_config(__file__)


@register_env("GateValveLargeFR3-v1", max_episode_steps=CONFIG["episode"]["max_episode_steps"])
class GateValveLargeFR3Env(HiveboardFR3Env):
    """Turn the large handwheel a full revolution by its central bolt or its rim, regrasping between segments."""

    config = CONFIG
    failure_keys = ("never_contacted", "stopped_short", "unwound", "timeout")
