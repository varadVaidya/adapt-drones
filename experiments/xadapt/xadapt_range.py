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

from adapt_drones.cfgs.config import *
from adapt_drones.utils.eval import phase1_eval, RMA_DATT_eval
from adapt_drones.utils.eval import paper_phase_1_eval, paper_RMA_DATT_eval
from adapt_drones.utils.git_utils import check_git_clean
from adapt_drones.utils.dynamics import CustomDynamics

from experiments.xadapt.scale_functions import *

import concurrent.futures
import tqdm as tqdm

# check_git_clean()  # TODO: see if needed in eval


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


def simulate_traj_rma(idx, seed, c, cfg):
    mean_error = 0
    rms_error = 0

    rng = np.random.default_rng(seed=seed)

    xadapt_dynamics = Dynamics(seed=rng, c=c)

    dynamics = CustomDynamics(
        arm_length=xadapt_dynamics.length_scale(),
        mass=xadapt_dynamics.mass_scale(),
        ixx=xadapt_dynamics.ixx_yy_scale(),
        iyy=xadapt_dynamics.ixx_yy_scale(),
        izz=xadapt_dynamics.izz_scale(),
        km_kf=xadapt_dynamics.torque_to_thrust(),
    )

    results = paper_RMA_DATT_eval(
        cfg=cfg, best_model=True, idx=idx, options=asdict(dynamics)
    )

    mean_error, rms_error = results[:2]

    return idx, seed, c, mean_error, rms_error


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

        c = np.linspace(0, 1, 16)
        seeds = np.arange(4551, 4551 + 16)
        idx = np.arange(0, 14)

        print(c)
        print(seeds)
        print(idx)

        num_lengths = len(c)
        num_seeds = len(seeds)
        num_idx = len(idx)

        traj_rma_eval = np.zeros((num_idx, num_seeds, num_lengths, 5))
        traj_rma_eval[:, :, :, :] = np.nan

        map_iterable = [
            (int(i), int(seed), float(_c), cfg)
            for i in idx
            for seed in seeds
            for _c in c
        ]
        print(len(map_iterable))

        with concurrent.futures.ProcessPoolExecutor(max_workers=6) as executor:

            results = list(
                tqdm.tqdm(
                    executor.map(
                        simulate_traj_rma,
                        *zip(*map_iterable),
                        chunksize=1,
                    ),
                    total=len(map_iterable),
                )
            )

        for result in results:
            _idx, _seed, _c, _mean_error, _rms_error = result
            idx_idx = np.where(idx == _idx)[0][0]
            seed_idx = np.where(seeds == _seed)[0][0]
            c_idx = np.where(c == _c)[0][0]

            traj_rma_eval[idx_idx, seed_idx, c_idx, :] = [
                _idx,
                _seed,
                _c,
                _mean_error,
                _rms_error,
            ]

        print(traj_rma_eval.shape)

        # # check for nans
        # print(np.argwhere(np.isnan(traj_rma_eval)))
        run_folder = f"experiments/xadapt/results-xadapt/{env_run[1]}/"
        prefix = "wind_" if cfg.wind_bool else "no_wind_"
        results_folder = run_folder

        os.makedirs(results_folder, exist_ok=True)

        np.save(results_folder + prefix + "xadapt_traj_rma.npy", traj_rma_eval)

        # traj_rma_eval = np.load(results_folder + "xadapt_traj_rma.npy")
        ## remove the seeds which performed worst across all scales, in
        # sense of mean error

        idx_sort_eval = np.argsort(np.mean(traj_rma_eval[:, :, :, 3], axis=2))
        sorted_traj_eval = traj_rma_eval[:, idx_sort_eval[0], :, :]

        # remove the top 3 and bottom 3 seeds
        traj_rma_eval = sorted_traj_eval[:, 3:-3, :, :]

        # compile the data into a csv file contain errors and std dev
        data_compile = np.zeros((traj_rma_eval.shape[0] * traj_rma_eval.shape[2], 7))

        # ^ 0: idx, 1: c,
        # ^ 3:  avg mean error, 4: std dev mean error,
        # ^ 4: avg rms error, 5: std dev rms error
        # ^ 6: crashed: num of seeds crashed

        data_compile[:, 0] = np.repeat(
            np.arange(traj_rma_eval.shape[0]), traj_rma_eval.shape[2]
        )  # tile idx
        data_compile[:, 1] = np.tile(
            traj_rma_eval[0, 0, :, 2], traj_rma_eval.shape[0]
        )  # tile c

        data_compile[:, 2] = np.mean(
            traj_rma_eval[:, :, :, 3],
            axis=1,
            where=~np.isinf(traj_rma_eval[:, :, :, 3]),
        ).reshape(
            traj_rma_eval.shape[0] * traj_rma_eval.shape[2]
        )  # avg mean error

        data_compile[:, 3] = np.std(
            traj_rma_eval[:, :, :, 3],
            axis=1,
            where=~np.isinf(traj_rma_eval[:, :, :, 3]),
        ).reshape(
            traj_rma_eval.shape[0] * traj_rma_eval.shape[2]
        )  # std dev mean error

        data_compile[:, 4] = np.mean(
            traj_rma_eval[:, :, :, 4],
            axis=1,
            where=~np.isinf(traj_rma_eval[:, :, :, 4]),
        ).reshape(
            traj_rma_eval.shape[0] * traj_rma_eval.shape[2]
        )  # avg rms error

        data_compile[:, 5] = np.std(
            traj_rma_eval[:, :, :, 4],
            axis=1,
            where=~np.isinf(traj_rma_eval[:, :, :, 4]),
        ).reshape(
            traj_rma_eval.shape[0] * traj_rma_eval.shape[2]
        )  # std dev rms error

        data_compile[:, 6] = np.sum(
            traj_rma_eval[:, :, :, 3] == np.inf,
            axis=1,
        ).reshape(
            traj_rma_eval.shape[0] * traj_rma_eval.shape[2]
        )  # crashed count

        print(data_compile.shape)
        print("Saving results to:")
        header = [
            "idx",
            "c",
            "mean_error",
            "std_dev_mean_error",
            "rms_error",
            "std_dev_rms_error",
            "crashed",
        ]
        np.savetxt(
            results_folder + prefix + "xadapt_traj_excel.csv",
            data_compile,
            delimiter=",",
            header=",".join(header),
            comments="",
        )

        import pandas as pd

        def get_error_info(df, idx_value, c_value):
            # Filter the row based on idx and c values
            row = df.loc[(df["idx"] == idx_value) & (df["c"] == c_value)]

            # If the row exists, round and concatenate mean_error and std_dev_mean_error
            if not row.empty:
                mean_error = round(row["mean_error"].values[0], 3)
                std_dev_mean_error = round(row["std_dev_mean_error"].values[0], 3)
                return f"{mean_error} \pm {std_dev_mean_error}"
            else:
                return "No match found"

        df = pd.read_csv(results_folder + prefix + "xadapt_traj_excel.csv")

        tex_table = []
        for idx_value in np.unique(df["idx"]):
            idx_row = []
            for c_value in np.unique(df["c"]):
                idx_row.append(get_error_info(df, idx_value, c_value))
            tex_table.append(idx_row)

        # convert tex_table to pandas dataframe
        # print(Dynamics(seed=0, c=np.unique(df["c"]), do_random=False).length_scale())
        tex_table = np.array(tex_table)
        tex_table = pd.DataFrame(
            tex_table,
            columns=np.round(
                Dynamics(seed=0, c=np.unique(df["c"]), do_random=False).length_scale(),
                3,
            ),
            index=np.unique(df["idx"]),
        )
        # save the table to a csv file
        tex_table.to_csv(results_folder + prefix + "xadapt_table.csv")
