import os
import random
import subprocess
from dataclasses import dataclass
from typing import Union

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import tyro

from adapt_drones.cfgs.config import *
from adapt_drones.utils.eval import custom_traj_eval
from adapt_drones.utils.git_utils import check_git_clean
from adapt_drones.utils.trajectory import (
    lissajous_3d_with_smooth_start,
    random_trajectory,
    random_straight_trajectory,
)

# check_git_clean()  # TODO: see if needed in eval


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "snowy-lake-170"
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

current_branch_name = (
    subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    .decode("utf-8")
    .strip()
)
print("Current branch name:", current_branch_name)
branch_name = "runs/" + cfg.experiment.grp_name + "/" + args.run_name

# checkout to the run tag
# subprocess.check_output(["git", "checkout", branch_name])

# reference_traj = lissajous_3d_with_smooth_start(
#     A=2.0,
#     B=2.5,
#     C=0.5,
#     a_w=3,
#     b_w=2,
#     delta=np.pi / 6,
#     time_per_loop=40.0,
#     total_time=120.0,
# )

reference_traj = random_straight_trajectory(seed=cfg.seed, total_time=30.0)
custom_traj_eval(cfg=cfg, best_model=True, reference_traj=reference_traj)


# return to the original branch
# subprocess.check_output(["git", "checkout", current_branch_name])
