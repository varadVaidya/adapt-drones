import os
import random
import time
import yaml
from dataclasses import asdict

import numpy as np
import torch
import gymnasium as gym
import tyro

from adapt_drones.cfgs.config import *
from adapt_drones.utils.learning import make_env
from adapt_drones.networks.ppo import ppo_train


@dataclass
class Args:
    env_id: str
    seed: int = 15092024
    wind_bool: bool = True
    agent: str = "RMA_DATT"


args = tyro.cli(Args)
cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    tests=True,
    agent=args.agent,
    wind_bool=args.wind_bool,
)

# set random seeds
random.seed(cfg.seed)
np.random.seed(cfg.seed)
torch.manual_seed(cfg.seed)
torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic


device = torch.device(
    "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
)

if cfg.experiment.track:
    import wandb

    run = wandb.init(
        project=cfg.experiment.wandb_project_name,
        group=cfg.experiment.grp_name,
        config=asdict(cfg),
        sync_tensorboard=True,
        settings=wandb.Settings(_disable_stats=True),
    )

    cfg.run_name = run.name

# env setup
envs = gym.vector.SyncVectorEnv(
    [make_env(cfg.env_id, cfg=cfg) for _ in range(cfg.learning.num_envs)]
)

ppo_train(args=cfg, envs=envs)
cfg = asdict(cfg)

with open("tests/outputs/ppo_test_cfg.yaml", "w") as file:
    yaml.dump(cfg, file)
