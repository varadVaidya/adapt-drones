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
import yaml
import wandb

from adapt_drones.cfgs.config import *
from adapt_drones.utils.learning import make_env
from adapt_drones.networks.ppo import ppo_train


@dataclass
class Args:
    env_id: str
    scale: bool = True
    seed: int = 15092024
    agent: str = "RMA_DATT"


def sweep(cfg, sweep_id):

    def main():
        try:
            run = wandb.init(
                group=cfg.experiment.grp_name,
                config=asdict(cfg),
                sync_tensorboard=True,
                settings=wandb.Settings(_disable_stats=True),
            )
            wandb.config.learning["batch_size"] = int(
                wandb.config.learning["num_envs"] * wandb.config.learning["num_steps"]
            )
            wandb.config.learning["minibatch_size"] = int(
                wandb.config.learning["batch_size"]
                // wandb.config.learning["num_minibatches"]
            )
            wandb.config.learning["num_iterations"] = (
                wandb.config.learning["total_timesteps"]
                // wandb.config.learning["batch_size"]
            )

            cfg.learning.__dict__.update(dict(wandb.config.learning))
            cfg.learning.batch_size = int(cfg.learning.batch_size)
            cfg.learning.minibatch_size = int(cfg.learning.minibatch_size)
            cfg.learning.num_iterations = (
                cfg.learning.total_timesteps // cfg.learning.batch_size
            )
            cfg.learning.update_epochs = int(cfg.learning.update_epochs)
            print(cfg.learning)
            cfg.run_name = run.name
            cfg.experiment.run_name = run.name
            cfg.experiment.track = True

            envs = gym.vector.SyncVectorEnv(
                [make_env(cfg.env_id, cfg=cfg) for _ in range(cfg.learning.num_envs)]
            )

            ppo_train(args=cfg, envs=envs)

        except Exception as e:
            import traceback

            traceback.print_exc()

    wandb.agent(
        sweep_id, main, count=100, project=cfg.experiment.wandb_project_name + "_sweep"
    )


if __name__ == "__main__":
    args = tyro.cli(Args)
    learning = Learning(total_timesteps=20_000_000)
    cfg = Config(
        env_id=args.env_id,
        seed=args.seed,
        scale=args.scale,
        agent=args.agent,
        learning=learning,
    )

    # set random seeds
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    torch.backends.cudnn.deterministic = cfg.learning.torch_deterministic

    device = torch.device(
        "cuda" if torch.cuda.is_available() and cfg.learning.cuda else "cpu"
    )
    sweep_ids = yaml.load(
        open("adapt_drones/utils/sweep_ids.yaml", "r"), Loader=yaml.FullLoader
    )
    # convert None string to None
    sweep_ids = {k: None if v == "None" else v for k, v in sweep_ids.items()}

    sweep_id = sweep_ids[cfg.experiment.grp_name]

    if sweep_id is None:
        print("Creating new sweep")
        sweep_dict = yaml.load(
            open("adapt_drones/utils/sweep.yaml", "r"), Loader=yaml.FullLoader
        )
        print(sweep_dict)
        sweep_id = wandb.sweep(
            sweep=sweep_dict,
            project=cfg.experiment.wandb_project_name + "_sweep",
        )
        # update sweep_ids.yaml
        sweep_ids[cfg.experiment.grp_name] = sweep_id
        yaml.dump(
            sweep_ids,
            open("adapt_drones/utils/sweep_ids.yaml", "w"),
            default_flow_style=False,
        )

    print("Sweep ID: ", sweep_id)

    sweep(cfg, sweep_id)
