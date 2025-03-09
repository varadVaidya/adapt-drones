import os
import random
import time
import yaml
from dataclasses import asdict
import subprocess

import numpy as np
import torch
import gymnasium as gym
import tyro

from adapt_drones.cfgs.config import *
from adapt_drones.utils.learning import make_env

from adapt_drones.networks.adapt_net import adapt_train_datt_rma
from adapt_drones.utils.git_utils import check_git_clean

check_git_clean()


@dataclass
class Args:
    env_id: str
    run_name: str
    scale: bool = True
    seed: int = 15092024
    agent: str = "RMA_DATT"


args = tyro.cli(Args)
learning = Learning(
    init_lr=2e-4,
    anneal_lr=False,
    num_envs=128,
    total_timesteps=5_000_000,
)
kwargs = {"learning": learning}
cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    run_name=args.run_name,
    scale=args.scale,
    agent=args.agent,
    **kwargs,
)

# set random seeds
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

device = torch.device(
    "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
)

envs = gym.vector.SyncVectorEnv(
    [make_env(cfg.env_id, cfg=cfg) for _ in range(cfg.learning.num_envs)]
)

adapt_train_datt_rma(cfg, envs)
