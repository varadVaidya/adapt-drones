import os
import random
import subprocess
from dataclasses import dataclass, asdict
from typing import Union

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import tyro
import pandas as pd

from adapt_drones.cfgs.config import *
from adapt_drones.utils.eval import paper_RMA_DATT_eval, paper_custom_traj_eval
from adapt_drones.utils.trajectory import random_trajectory

import concurrent.futures
import tqdm as tqdm


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "lemon-armadillo-68"
    seed: int = 15092024
    agent: str = "RMA_DATT"
    scale: bool = True
    idx: Union[int, None] = None
    wind_bool: bool = True


args = tyro.cli(Args)
cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    eval=True,
    run_name=args.run_name,
    agent=args.agent,
    scale=args.scale,
    wind_bool=args.wind_bool,
)


def evaluate_per_seed_per_scale(seed, scale, cfg, total_time):
    cfg.seed = seed
    cfg.environment.scale_lengths = scale
    cfg.scale.scale_lengths = scale

    trajectory = random_trajectory(seed=seed, total_time=total_time)

    results = paper_custom_traj_eval(
        cfg=cfg, best_model=True, reference_traj=trajectory
    )

    return total_time, seed, scale[0], results[0], results[1]


if __name__ == "__main__":
    env_runs = [
        ["traj_v3", "earthy-snowball-77", True],
        # ["traj_v3", "fine-universe-76", True],
    ]
    for env_run in env_runs:
        args = Args(env_id=env_run[0], run_name=env_run[1], wind_bool=env_run[2])

        print(f"Running evaluation for {args.run_name} with wind: {args.wind_bool}")

        cfg = Config(
            env_id=args.env_id,
            seed=args.seed,
            eval=True,
            run_name=args.run_name,
            agent=args.agent,
            scale=args.scale,
            wind_bool=args.wind_bool,
        )

        seed = -1
        seed = seed if seed > 0 else random.randint(0, 2**32 - 1)
        scale = 0.16
        total_time = 30

        print("Seed: ", seed)
        print("Scale: ", scale)
        print("Total Time: ", total_time)

        print(evaluate_per_seed_per_scale(seed, [scale, scale], cfg, total_time))
