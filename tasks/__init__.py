# Registers FR3 HiveBoard environments and maps task names to their simulation classes.
# Created: 2026-09-14

from tasks.ball_valve.task import BallValveFR3Env
from tasks.ball_valve_ring.task import BallValveRingFR3Env
from tasks.button_cover.task import ButtonCoverFR3Env
from tasks.circuit_breaker.task import CircuitBreakerFR3Env
from tasks.drawer.task import DrawerFR3Env
from tasks.gate_valve_large.task import GateValveLargeFR3Env
from tasks.gate_valve_small.task import GateValveSmallFR3Env
from tasks.key_lock.task import KeyLockFR3Env
from tasks.light_bulb.task import LightBulbFR3Env
from tasks.peg_insertion.task import PegInsertionFR3Env
from tasks.shock_absorber.task import ShockAbsorberFR3Env
from tasks.thread_m8.task import ThreadM8FR3Env
from tasks.thread_m30.task import ThreadM30FR3Env

TASKS = {
    "ball_valve": "BallValveFR3-v1",
    "ball_valve_ring": "BallValveRingFR3-v1",
    "circuit_breaker": "CircuitBreakerFR3-v1",
    "gate_valve_small": "GateValveSmallFR3-v1",
    "button_cover": "ButtonCoverFR3-v1",
    "key_lock": "KeyLockFR3-v1",
    "gate_valve_large": "GateValveLargeFR3-v1",
    "drawer": "DrawerFR3-v1",
    "peg_insertion": "PegInsertionFR3-v1",
    "thread_m8": "ThreadM8FR3-v1",
    "thread_m30": "ThreadM30FR3-v1",
    "light_bulb": "LightBulbFR3-v1",
    "shock_absorber": "ShockAbsorberFR3-v1",
}
ENV_CLASSES = {
    "ball_valve": BallValveFR3Env,
    "ball_valve_ring": BallValveRingFR3Env,
    "circuit_breaker": CircuitBreakerFR3Env,
    "gate_valve_small": GateValveSmallFR3Env,
    "button_cover": ButtonCoverFR3Env,
    "key_lock": KeyLockFR3Env,
    "gate_valve_large": GateValveLargeFR3Env,
    "drawer": DrawerFR3Env,
    "peg_insertion": PegInsertionFR3Env,
    "thread_m8": ThreadM8FR3Env,
    "thread_m30": ThreadM30FR3Env,
    "light_bulb": LightBulbFR3Env,
    "shock_absorber": ShockAbsorberFR3Env,
}
