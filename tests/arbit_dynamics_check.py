import os

os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass, asdict
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import tyro

import adapt_drones
import adapt_drones.envs
from adapt_drones.cfgs.config import *
from adapt_drones.utils.dynamics import CustomDynamics


@dataclass
class Args:
    env_id: str
    seed: int = 15092024
    wind_bool: bool = True
    agent: str = "RMA_DATT"


args = tyro.cli(Args)

cfg = Config(
    env_id=args.env_id, seed=args.seed, agent=args.agent, wind_bool=args.wind_bool
)

env = gym.make(args.env_id, cfg=cfg, record=False)
env = gym.wrappers.FlattenObservation(env)
env = gym.wrappers.RecordEpisodeStatistics(env)

obs, _ = env.reset(
    seed=15092024,
    options=asdict(
        CustomDynamics(arm_length=0.2, mass=1.0, ixx=0.1, iyy=0.1, izz=0.1, km_kf=0.1)
    ),
)

for _ in range(600):
    action = env.action_space.sample()
    env.step(action)

print(env.unwrapped.model.body_mass)
print("--------------------")
print(env.unwrapped.model.body_inertia)
print("--------------------")
print(env.unwrapped.model.dof_invweight0)
