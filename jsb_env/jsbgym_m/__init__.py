import gymnasium as gym
import enum
import os
import sys
current_file_path = os.path.abspath(__file__)
parent_dir = os.path.join(os.path.dirname(current_file_path), "..")
if parent_dir not in sys.path:
    sys.path.append(parent_dir)
from jsbgym_m.tasks import Task, HeadingControlTask, TurnHeadingControlTask, Shaping
from jsbgym_m.task_tracking import TrackingTask, Opponent
from jsbgym_m.aircraft import Aircraft, c172, f16
from jsbgym_m import utils

"""
This script registers all combinations of task, aircraft, shaping settings
 etc. with Farama Foundation Gymnasium so that they can be instantiated with a gym.make(id)
 command.

The jsbgym_m.Envs enum stores all registered environments as members with
 their gym id string as value. This allows convenient autocompletion and value
 safety. To use do:
       env = gym.make(jsbgym_m.Envs.desired_environment.value)
"""

for env_id, (
    plane,
    task,
    shaping,
    enable_flightgear,
) in utils.get_env_id_kwargs_map().items():
    # 添加检查，避免重复注册环境
    if env_id in gym.envs.registry:
        continue
        
    if issubclass(task, TrackingTask):
        # opponent = task("stage1",1,f16).opponent
        if enable_flightgear:
            entry_point = "jsbgym_m.environment:DoubleJsbSimEnv" #if opponent == "jsbsim" else "jsbgym_m.environment:JsbSimEnv"
        else:
            entry_point = "jsbgym_m.environment:NoFGDoubleJsbSimEnv" #if opponent == "jsbsim" else "jsbgym_m.environment:NoFGJsbSimEnv"
    else:
        if enable_flightgear:
            entry_point = "jsbgym_m.environment:JsbSimEnv"
        else:
            entry_point = "jsbgym_m.environment:NoFGJsbSimEnv"
    kwargs = dict(aircraft=plane, task_type=task, shaping=shaping)
    gym.envs.registration.register(id=env_id, entry_point=entry_point, kwargs=kwargs)

# make an Enum storing every Gym-JSBSim environment ID for convenience and value safety
Envs = enum.Enum.__call__(
    "Envs",
    [
        (utils.AttributeFormatter.translate(env_id), env_id)
        for env_id in utils.get_env_id_kwargs_map().keys()
    ],
)
