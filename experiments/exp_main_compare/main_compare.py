import os
import random
import subprocess
from dataclasses import dataclass, asdict
import concurrent.futures
import multiprocessing as mp
from itertools import repeat
import time
import tqdm

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import tyro
import pandas as pd

from adapt_drones.cfgs.config import *
from adapt_drones.cfgs.environment_cfg import *
from adapt_drones.utils.eval import paper_fig_RMA_DATT_eval
from adapt_drones.utils.dynamics import CustomDynamics


@dataclass
class Args:
    env_id: str
    run_name: str
    seed: int = 4551
    agent: str = "RMA_DATT"
    scale: bool = True
    wind_bool: bool = True


def simulate_traj(idx, cfg):

    dynamics = CustomDynamics(
        arm_length=0.177, mass=0.985, ixx=4e-3, iyy=8e-3, izz=12e-3, km_kf=0.014
    )

    results = paper_fig_RMA_DATT_eval(
        cfg=cfg, best_model=True, idx=idx, options=asdict(dynamics)
    )

    return results


if __name__ == "__main__":
    env_runs = [
        ["traj_v3", "snowy-lake-170", True],
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
        cfg.environment.eval_trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/exp_eval_array.npy"
        )

               
        idx = np.arange(0,4)

        print(idx)

        results = []
        for i in idx:
            print(f"Running evaluation for trajectory {i}")
            result = simulate_traj(i, cfg)
            results.append(result)
            # print(result[-2], result[-1])

        # the shape of results is 2 x 24632 x 3
        np_results = np.empty((4, 3, 24632, 3))
        np_results[:] = np.nan

        for i, result in enumerate(results):
            print(result.shape)
            np_results[i, :, : len(result[1]), :] = result

        run_folder = f"experiments/exp_main_compare/rma_datt/"
        os.makedirs(run_folder, exist_ok=True)

        np.save(f"{run_folder}{args.run_name}_results.npy", np_results)
