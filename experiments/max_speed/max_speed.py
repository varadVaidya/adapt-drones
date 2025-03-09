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
from adapt_drones.utils.trajectory import lemniscate_trajectory, loop_trajectory

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


def evaluate_per_seed_per_scale(seed, scale, cfg, max_speed):
    cfg.seed = seed
    cfg.environment.scale_lengths = scale
    cfg.scale.scale_lengths = scale

    trajectory = lemniscate_trajectory(
        discretization_dt=0.01,
        radius=5,
        z=1,
        clockwise=True,
        lin_acc=0.25,
        yawing=False,
        v_max=max_speed,
    )

    # trajectory = loop_trajectory(
    #     discretization_dt=0.01,
    #     radius=5,
    #     z=1,
    #     clockwise=True,
    #     lin_acc=0.25,
    #     yawing=False,
    #     v_max=max_speed,
    # )

    results = paper_custom_traj_eval(
        cfg=cfg, best_model=True, reference_traj=trajectory
    )

    return max_speed, seed, scale[0], results[0], results[1]


if __name__ == "__main__":
    env_runs = [
        ["traj_v3", "snowy-lake-170", True],
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

        c = np.linspace(0.05, 0.22, 16)
        seeds = np.arange(4551, 4551 + 20)
        sc_list = [[i, i] for i in c]
        max_speeds = np.arange(1.0, 10.0, step=0.5)
        trajectory = "lemniscate"
        print("Max speeds:", max_speeds)

        print(c)
        print(seeds)
        print(max_speeds)

        num_lengths = len(c)
        num_seeds = len(seeds)
        num_speeds = len(max_speeds)

        # cfg.environment.wind_bool = False
        # cfg.environment.max_wind = 0.0
        # cfg.environment.wind_speed = [0.0, 0.0]

        map_iterable = [
            (int(seed), list(scale), cfg, max_speed)
            for seed in seeds
            for scale in sc_list
            for max_speed in max_speeds
        ]

        print("\t", len(map_iterable))

        # print(evaluate_per_seed_per_scale(4551, [0.15, 0.15], cfg, 5.0))

        traj_speed_eval = np.zeros((num_speeds, num_seeds, num_lengths, 5))
        traj_speed_eval[:, :, :, :] = np.nan

        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:

            results = list(
                tqdm.tqdm(
                    executor.map(
                        evaluate_per_seed_per_scale,
                        *zip(*map_iterable),
                        chunksize=1,
                    ),
                    total=len(map_iterable),
                )
            )

        for result in results:
            _speed, _seed, _scale, _mean_error, _rms_error = result
            _speed_idx = np.where(max_speeds == _speed)[0][0]
            _scale_idx = np.where(c == _scale)[0][0]
            _seed_idx = np.where(seeds == _seed)[0][0]

            traj_speed_eval[_speed_idx, _seed_idx, _scale_idx, :] = [
                _speed,
                _seed,
                _scale,
                _mean_error,
                _rms_error,
            ]

        run_folder = f"experiments/max_speed/results-speed/{trajectory}/{env_run[1]}/"
        prefix = "wind_" if cfg.wind_bool else "no_wind_"
        results_folder = run_folder

        os.makedirs(results_folder, exist_ok=True)

        np.save(results_folder + prefix + "traj_speed_rma.npy", traj_speed_eval)
