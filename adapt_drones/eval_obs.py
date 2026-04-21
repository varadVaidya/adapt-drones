import os
import subprocess
from dataclasses import dataclass
from typing import Union

os.environ["MUJOCO_GL"] = "egl"

import tyro

from adapt_drones.cfgs.config import *
from adapt_drones.utils.eval_obs import phase1_obs_eval, RMA_DATT_obs_eval


@dataclass
class Args:
    env_id: str = "traj_obs_v3"
    run_name: str = ""
    seed: int = 15092024
    agent: str = "RMA_DATT_Safe"
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
    wandb_project="adapt-safe",
)

current_branch_name = (
    subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"])
    .decode("utf-8")
    .strip()
)
print("Current branch name:", current_branch_name)
branch_name = "runs/" + cfg.experiment.grp_name + "/" + args.run_name

subprocess.check_output(["git", "checkout", branch_name])

phase1_obs_eval(cfg=cfg, best_model=True, idx=args.idx)

RMA_DATT_obs_eval(cfg=cfg, best_model=True, idx=args.idx)

subprocess.check_output(["git", "checkout", current_branch_name])
