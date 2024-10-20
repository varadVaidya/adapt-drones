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
from adapt_drones.utils.eval import paper_phase_1_eval, paper_RMA_DATT_eval

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


def evaluate_per_seed_per_scale(seed, scale, cfg, idx):
    cfg.seed = seed
    cfg.environment.scale_lengths = scale
    cfg.scale.scale_lengths = scale

    results = paper_RMA_DATT_eval(cfg=cfg, best_model=True, idx=idx)

    return idx, seed, scale[0], results[0], results[1]


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

        # cfg.environment.max_wind = 12
        # cfg.environment.wind_speed = [12, 12]

        c = np.linspace(0.05, 0.22, 16)
        seeds = np.arange(4551, 4551 + 16)
        idx = np.array([2, 5])
        sc_list = [[i, i] for i in c]
        wind_speeds = np.arange(3, 17, step=2)

        print(c)
        print(seeds)
        print(idx)
        print(wind_speeds)

        num_lengths = len(c)
        num_seeds = len(seeds)
        num_idx = len(idx)

        for i in range(len(wind_speeds)):
            max_speed = wind_speeds[i]
            cfg.environment.max_wind = max_speed
            cfg.environment.wind_speed = [max_speed, max_speed]

            print("\t Running evaluation for wind speed: ", max_speed)

            map_iterable = [
                (int(seed), list(scale), cfg, int(i))
                for seed in seeds
                for scale in sc_list
                for i in idx
            ]

            print("\t", len(map_iterable))

            traj_wind_eval = np.zeros((num_idx, num_seeds, num_lengths, 5))
            traj_wind_eval[:, :, :, :] = np.nan

            with concurrent.futures.ProcessPoolExecutor(max_workers=12) as executor:

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
                _idx, _seed, _scale, _mean_error, _rms_error = result
                _idx_idx = np.where(idx == _idx)[0][0]
                _seed_idx = np.where(seeds == _seed)[0][0]
                _scale_idx = np.where(c == _scale)[0][0]

                traj_wind_eval[_idx_idx, _seed_idx, _scale_idx, :] = [
                    _idx,
                    _seed,
                    _scale,
                    _mean_error,
                    _rms_error,
                ]

            run_folder = f"experiments/max_wind/results-wind/{env_run[1]}/"
            prefix = "wind_" if cfg.wind_bool else "no_wind_"
            prefix += f"{max_speed}_"
            results_folder = run_folder

            os.makedirs(results_folder, exist_ok=True)

            np.save(results_folder + prefix + "wind_traj_rma.npy", traj_wind_eval)

            for i in range(len(traj_wind_eval)):
                idx_sort_eval = np.argsort(np.mean(traj_wind_eval[i, :, :, 3], axis=1))
                traj_wind_eval[i, :, :, :] = traj_wind_eval[i, idx_sort_eval, :, :]

            # remove top 3 and bottom 3 seeds
            traj_wind_eval = traj_wind_eval[:, 3:-3, :, :]

            # compile the data into a csv file contain errors and std dev
            data_compile = np.zeros(
                (traj_wind_eval.shape[0] * traj_wind_eval.shape[2], 7)
            )

            # ^ 0: idx, 1: c,
            # ^ 3:  avg mean error, 4: std dev mean error,
            # ^ 4: avg rms error, 5: std dev rms error
            # ^ crash rate

            data_compile[:, 0] = np.repeat(
                np.arange(traj_wind_eval.shape[0]), traj_wind_eval.shape[2]
            )  # tile idx

            data_compile[:, 1] = np.tile(
                traj_wind_eval[0, 0, :, 2], traj_wind_eval.shape[0]
            )  # tile c

            data_compile[:, 2] = np.mean(
                traj_wind_eval[:, :, :, 3],
                axis=1,
                where=traj_wind_eval[:, :, :, 3] < 0.5,
            ).reshape(
                traj_wind_eval.shape[0] * traj_wind_eval.shape[2]
            )  # avg mean error

            data_compile[:, 3] = np.std(
                traj_wind_eval[:, :, :, 3],
                axis=1,
                where=traj_wind_eval[:, :, :, 3] < 0.5,
            ).reshape(
                traj_wind_eval.shape[0] * traj_wind_eval.shape[2]
            )  # std dev mean error

            data_compile[:, 4] = np.mean(
                traj_wind_eval[:, :, :, 4],
                axis=1,
                where=traj_wind_eval[:, :, :, 4] < 0.5,
            ).reshape(
                traj_wind_eval.shape[0] * traj_wind_eval.shape[2]
            )  # avg rms error

            data_compile[:, 5] = np.std(
                traj_wind_eval[:, :, :, 4],
                axis=1,
                where=traj_wind_eval[:, :, :, 4] < 0.5,
            ).reshape(
                traj_wind_eval.shape[0] * traj_wind_eval.shape[2]
            )  # std dev rms error

            # crash rate or success rate is calculated as the number of time,
            # the mean error of the idx-scale pair is less than 0.2 across all seeds
            data_compile[:, 6] = (
                np.sum(traj_wind_eval[:, :, :, 3] < 0.5, axis=1).reshape(
                    traj_wind_eval.shape[0] * traj_wind_eval.shape[2]
                )
                / traj_wind_eval.shape[1]
            )

            print(data_compile.shape)

            print("Saving data_compile to: ", run_folder + prefix + "traj_excel.csv")

            header = [
                "idx",
                "c",
                "mean_error",
                "std_dev_mean_error",
                "rms_error",
                "std_dev_rms_error",
                "success_rate",
            ]

            # np.savetxt(
            #     run_folder + prefix + "traj_excel.csv",
            #     data_compile,
            #     delimiter=",",
            #     header=",".join(header),
            #     comments="",
            # )

            def get_error_info(df, idx_value, c_value):
                # Filter the row based on idx and c values
                row = df.loc[(df["idx"] == idx_value) & (df["c"] == c_value)]

                # If the row exists, round and concatenate mean_error and std_dev_mean_error
                if not row.empty:
                    mean_error = round(row["mean_error"].values[0], 4)
                    std_dev_mean_error = round(row["std_dev_mean_error"].values[0], 4)
                    return f"{mean_error} \pm {std_dev_mean_error}"
                else:
                    return "No match found"

            def get_success_rate(df, idx_value, c_value):
                # Filter the row based on idx and c values
                row = df.loc[(df["idx"] == idx_value) & (df["c"] == c_value)]

                # If the row exists, round and concatenate mean_error and std_dev_mean_error
                if not row.empty:
                    success_rate = round(row["success_rate"].values[0], 4)
                    return success_rate
                else:
                    return "No match found"

            df = pd.read_csv(run_folder + prefix + "traj_excel.csv")

            tex_table = []
            for idx_value in np.unique(df["idx"]):
                idx_row = []
                for c_value in np.unique(df["c"]):
                    idx_row.append(get_error_info(df, idx_value, c_value))
                    idx_row.append(get_success_rate(df, idx_value, c_value))
                tex_table.append(idx_row)

            # convert tex_table to pandas dataframe
            tex_table = np.array(tex_table)
            tex_table = pd.DataFrame(
                tex_table,
                columns=np.repeat(np.unique(df["c"]), 2),
                index=np.unique(df["idx"]),
            )
            # save the table to a csv file
            tex_table.to_csv(run_folder + prefix + "eval_table.csv")
