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
from adapt_drones.utils.git_utils import check_git_clean
from adapt_drones.utils.dynamics import CustomDynamics

# check_git_clean()  # TODO: see if needed in eval


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "laced-fire-32"
    seed: int = 15092024
    agent: str = "RMA_DATT"
    scale: bool = True
    idx: Union[int, None] = None
    wind_bool: bool = False


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

dynamics = CustomDynamics(
    arm_length=0.058, mass=0.267, ixx=259e-6, iyy=228e-6, izz=285e-6, km_kf=0.008
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

# phase 1 eval
phase1_eval(cfg=cfg, best_model=True, idx=args.idx, options=asdict(dynamics))

# rma eval
RMA_DATT_eval(cfg=cfg, best_model=True, idx=args.idx, options=asdict(dynamics))

# return to the original branch
# subprocess.check_output(["git", "checkout", current_branch_name])
