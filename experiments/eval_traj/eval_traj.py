import os
import random
import subprocess
from dataclasses import dataclass
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
from adapt_drones.utils.eval import paper_phase_1_eval, paper_RMA_DATT_eval
from adapt_drones.utils.git_utils import check_git_clean


@dataclass
class Args:
    env_id: str
    run_name: str
    seed: int = 4551
    agent: str = "RMA_DATT"
    scale: bool = True
    wind_bool: bool = True


def evaluate_per_seed_per_scale(seed, scale, cfg, idx):
    rma_datt_scale_results = np.zeros(8)
    cfg.seed = seed
    cfg.environment.scale_lengths = scale
    cfg.scale.scale_lengths = scale

    results = paper_RMA_DATT_eval(cfg=cfg, best_model=True, idx=idx)

    rma_datt_scale_results[0] = scale[0]
    rma_datt_scale_results[1] = seed
    rma_datt_scale_results[2:] = results

    return idx, seed, scale[0], results[0], results[1]


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

        c = np.linspace(0.05, 0.22, 16)
        seeds = np.arange(4551, 4551 + 16)
        idx = np.arange(0, 14)

        sc_list = [[i, i] for i in c]

        print(sc_list)
        print(seeds)
        print(idx)

        num_lengths = len(c)
        num_seeds = len(seeds)
        num_idx = len(idx)

        map_iterable = [
            (int(seed), list(scale), cfg, int(i))
            for seed in seeds
            for scale in sc_list
            for i in idx
        ]

        print(len(map_iterable))

        # # print(evaluate_per_seed_per_scale(*map_iterable[0]))

        # # for i in tqdm.tqdm(range(len(map_iterable))):
        # #     print(evaluate_per_seed_per_scale(*map_iterable[i]))

        traj_scale_eval = np.zeros((num_idx, num_seeds, num_lengths, 5))
        traj_scale_eval[:, :, :, :] = np.nan

        prefix = "wind_" if env_run[2] else "no_wind_"

        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:
            results = list(
                tqdm.tqdm(
                    executor.map(
                        evaluate_per_seed_per_scale, *zip(*map_iterable), chunksize=1
                    ),
                    total=len(map_iterable),
                )
            )

        for result in results:
            _idx, _seed, _c, _mean_error, _rms_error = result
            idx_idx = np.where(idx == _idx)[0][0]
            seed_idx = np.where(seeds == _seed)[0][0]
            c_idx = np.where(c == _c)[0][0]

            traj_scale_eval[idx_idx, seed_idx, c_idx] = [
                _idx,
                _seed,
                _c,
                _mean_error,
                _rms_error,
            ]

        print(traj_scale_eval.shape)

        # # check for nans
        # print(np.argwhere(np.isnan(traj_scale_eval)))

        run_folder = f"experiments/eval_traj/results-scale/{env_run[1]}/"

        os.makedirs(run_folder, exist_ok=True)

        np.save(run_folder + prefix + "traj_scale_eval.npy", traj_scale_eval)

        # remove the seeds which performed worst across all scales, in
        # sense of mean error
        traj_scale_eval = np.load(run_folder + prefix + "traj_scale_eval.npy")
        print(traj_scale_eval.shape)

        idx_sort_eval = np.argsort(traj_scale_eval[:, :, :, 3], axis=2)
        idx_sort_eval = idx_sort_eval[:, :, :, np.newaxis]
        sorted_traj_eval = np.take_along_axis(
            traj_scale_eval[:, :, :, :], idx_sort_eval, axis=2
        )
        print(sorted_traj_eval.shape)

        # remove top 3 and bottom 3 seeds
        traj_scale_eval = sorted_traj_eval[:, 3:-3, :, :]
        # traj_scale_eval = sorted_t

        print(traj_scale_eval.shape)

        # compile the data into a csv file contain errors and std dev
        data_compile = np.zeros(
            (traj_scale_eval.shape[0] * traj_scale_eval.shape[2], 7)
        )

        # ^ 0: idx, 1: c,
        # ^ 3:  avg mean error, 4: std dev mean error,
        # ^ 4: avg rms error, 5: std dev rms error
        # ^ crash rate

        data_compile[:, 0] = np.repeat(
            np.arange(traj_scale_eval.shape[0]), traj_scale_eval.shape[2]
        )  # tile idx

        data_compile[:, 1] = np.tile(
            traj_scale_eval[0, 0, :, 2], traj_scale_eval.shape[0]
        )  # tile c

        data_compile[:, 2] = np.mean(
            traj_scale_eval[:, :, :, 3], axis=1, where=traj_scale_eval[:, :, :, 3] < 0.2
        ).reshape(
            traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
        )  # avg mean error

        data_compile[:, 3] = np.std(
            traj_scale_eval[:, :, :, 3], axis=1, where=traj_scale_eval[:, :, :, 3] < 0.5
        ).reshape(
            traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
        )  # std dev mean error

        data_compile[:, 4] = np.mean(
            traj_scale_eval[:, :, :, 4], axis=1, where=traj_scale_eval[:, :, :, 4] < 0.5
        ).reshape(
            traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
        )  # avg rms error

        data_compile[:, 5] = np.std(
            traj_scale_eval[:, :, :, 4], axis=1, where=traj_scale_eval[:, :, :, 4] < 0.5
        ).reshape(
            traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
        )  # std dev rms error

        # crash rate or success rate is calculated as the number of time,
        # the mean error of the idx-scale pair is less than 0.2 across all seeds
        data_compile[:, 6] = (
            np.sum(traj_scale_eval[:, :, :, 3] < 0.5, axis=1).reshape(
                traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
            )
            / traj_scale_eval.shape[1]
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

        np.savetxt(
            run_folder + prefix + "traj_excel.csv",
            data_compile,
            delimiter=",",
            header=",".join(header),
            comments="",
        )

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
        tex_table.to_csv(run_folder + prefix + "traj_eval_table.csv")
