import os

os.environ["MUJOCO_GL"] = "egl"

from dataclasses import dataclass
import numpy as np
import gymnasium as gym
from gymnasium.utils.env_checker import check_env

import tyro

import adapt_drones
import adapt_drones.envs
from adapt_drones.cfgs.config import *


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
print(cfg.environment)
env = gym.make(args.env_id, cfg=cfg, record=True)
env = gym.wrappers.FlattenObservation(env)
env = gym.wrappers.RecordEpisodeStatistics(env)

obs, _ = env.reset(seed=15092024)

for _ in range(600):
    action = env.action_space.sample()
    env.step(action)

env.unwrapped.vidwrite("tests/outputs/env_check.mp4")
env.unwrapped.renderer.close()
