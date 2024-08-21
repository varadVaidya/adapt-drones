import os
import random
import subprocess
from dataclasses import dataclass

os.environ["MUJOCO_GL"] = "egl"

import numpy as np
import torch
import gymnasium as gym
import tyro

from adapt_drones.cfgs.config import *
from adapt_drones.utils.eval import phase1_eval
from adapt_drones.utils.git_utils import check_git_clean

# check_git_clean()  # TODO: see if needed in eval


@dataclass
class Args:
    env_id: str
    run_name: str
    seed: int = 15092024


args = tyro.cli(Args)
cfg = Config(env_id=args.env_id, seed=args.seed, eval=True, run_name=args.run_name)

current_branch_name = (
    subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    .decode("utf-8")
    .strip()
)
print("Current branch name:", current_branch_name)
branch_name = "runs/" + cfg.experiment.grp_name + "/" + args.run_name

# checkoout to the branch
# subprocess.check_output(["git", "checkout", branch_name])

# phase 1 eval
phase1_eval(cfg=cfg, best_model=True)

# return to the original branch
# subprocess.check_output(["git", "checkout", current_branch_name])
