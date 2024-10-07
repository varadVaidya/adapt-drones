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
    rma_datt_scale_results = np.zeros(9)
    cfg.seed = seed
    cfg.environment.scale_lengths = scale
    cfg.scale.scale_lengths = scale

    results = paper_RMA_DATT_eval(
        cfg=cfg, best_model=True, idx=idx, return_traj_len=True
    )
    rma_datt_scale_results[0] = scale[0]
    rma_datt_scale_results[1] = seed
    rma_datt_scale_results[2:] = results

    return idx, seed, scale[0], results[0], results[1]


if __name__ == "__main__":
    env_run = ["traj_v3", "lemon-armadillo-68", True]
    args = Args(env_id=env_run[0], run_name=env_run[1], wind_bool=env_run[2])

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
    idx = np.arange(0, 13)
    sc_list = [[i, i] for i in c]

    print(sc_list)
    print(seeds)
    print(idx)

    num_lengths = len(c)
    num_seeds = len(seeds)
    num_idx = len(idx)

    map_iterable = [
        (int(seed), scale, cfg, i) for seed in seeds for scale in sc_list for i in idx
    ]

    print(len(map_iterable))

    print(evaluate_per_seed_per_scale(*map_iterable[0]))

    # for i in tqdm.tqdm(range(len(map_iterable))):
    #     print(evaluate_per_seed_per_scale(*map_iterable[i]))

    traj_scale_eval = np.zeros((num_idx, num_seeds, num_lengths, 5))
    traj_scale_eval[:, :, :, :] = np.nan

    prefix = "wind" if env_run[2] else "no_wind"

    with concurrent.futures.ProcessPoolExecutor(
        max_workers=8,
        mp_context=mp.get_context("spawn"),
    ) as executor:
        results = list(
            tqdm.tqdm(
                executor.map(evaluate_per_seed_per_scale, *zip(*map_iterable)),
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

    # check for nans
    print(np.argwhere(np.isnan(traj_scale_eval)))

    run_folder = "experiments/eval_traj/results-scale/"

    os.makedirs(run_folder, exist_ok=True)

    np.save(run_folder + prefix + "traj_scale_eval.npy", traj_scale_eval)

    # remove the seeds which performed worst across all scales, in
    # sense of mean error

    idx_sort_eval = np.argsort(traj_scale_eval[:, :, :, 3], axis=2)
    sorted_traj_eval = traj_scale_eval[:, idx_sort_eval[0], :, :]

    # remove top 3 and bottom 3 seeds
    # traj_scale_eval = sorted_traj_eval[:, 3:-3, :, :]

    # compile the data into a csv file contain errors and std dev
    data_compile = np.zeros((traj_scale_eval.shape[0] * traj_scale_eval.shape[2], 6))

    # ^ 0: idx, 1: c,
    # ^ 3:  avg mean error, 4: std dev mean error,
    # ^ 4: avg rms error, 5: std dev rms error

    data_compile[:, 0] = np.repeat(
        np.arange(traj_scale_eval.shape[0]), traj_scale_eval.shape[2]
    )  # tile idx

    data_compile[:, 1] = np.tile(
        traj_scale_eval[0, 0, :, 2], traj_scale_eval.shape[0]
    )  # tile c

    data_compile[:, 2] = np.mean(traj_scale_eval[:, :, :, 3], axis=1).reshape(
        traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
    )  # avg mean error

    data_compile[:, 3] = np.std(traj_scale_eval[:, :, :, 3], axis=1).reshape(
        traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
    )  # std dev mean error

    data_compile[:, 4] = np.mean(traj_scale_eval[:, :, :, 4], axis=1).reshape(
        traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
    )  # avg rms error

    data_compile[:, 5] = np.std(traj_scale_eval[:, :, :, 4], axis=1).reshape(
        traj_scale_eval.shape[0] * traj_scale_eval.shape[2]
    )  # std dev rms error

    print(data_compile.shape)

    print("Saving data_compile to: ", run_folder + prefix + "mpc_traj_excel.csv")

    header = [
        "idx",
        "c",
        "mean_error",
        "std_dev_mean_error",
        "rms_error",
        "std_dev_rms_error",
    ]

    np.savetxt(
        run_folder + prefix + "mpc_traj_excel.csv",
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

    df = pd.read_csv(run_folder + prefix + "mpc_traj_excel.csv")

    tex_table = []
    for idx_value in np.unique(df["idx"]):
        idx_row = []
        for c_value in np.unique(df["c"]):
            idx_row.append(get_error_info(df, idx_value, c_value))
        tex_table.append(idx_row)

    # convert tex_table to pandas dataframe
    tex_table = np.array(tex_table)
    tex_table = pd.DataFrame(
        tex_table,
        columns=np.round(
            np.unique(df["c"]),
            3,
        ),
        index=np.unique(df["idx"]),
    )
    # save the table to a csv file
    tex_table.to_csv(run_folder + prefix + "mpc_table.csv")
