import random
import warnings
from dataclasses import dataclass

from adapt_drones.cfgs.experiment_cfg import Experiment
from adapt_drones.cfgs.environment_cfg import *
from adapt_drones.cfgs.scale_cfg import Scale
from adapt_drones.cfgs.learning_cfg import Learning
from adapt_drones.cfgs.network_cfg import Network


# master dataclass
@dataclass
class Config:
    eval: bool
    seed: int
    env_id: str
    tests: bool
    grp_name: str
    run_name: str
    agent: str
    wind_bool: bool
    warm_start: bool
    warm_model: [str, None]

    # sub dataclasses
    experiment: Experiment
    environment: [
        HoverAviaryv0Config,
        HoverAviaryv1Config,
        TrajAviaryv2Config,
        TrajAviaryv3Config,
        TrajAviaryPayv3Config,
    ]
    scale: Scale
    learning: Learning
    network: Network

    def __init__(
        self,
        eval=False,
        seed=1,
        scale=True,
        wind_bool=True,
        env_id="hover_v0",
        tests=False,
        grp_name="default",
        run_name="default",
        agent="AC",
        warm_start=False,
        warm_model=None,
        wandb_project="adapt-test-new",
        **kwargs,
    ):

        track = not tests

        if not eval and seed == -1:
            warnings.warn("Eval is set to False, and seed is arbitrary")

        seed = seed if seed != -1 else random.randint(0, 2**32 - 1)

        self.eval = eval
        self.seed = seed
        self.env_id = env_id
        self.tests = tests
        self.grp_name = grp_name
        self.run_name = run_name
        self.agent = agent
        self.warm_start = warm_start
        self.warm_model = warm_model
        self.wind_bool = wind_bool

        if not tests:
            grp_name = env_id + "-" + agent
            self.grp_name = grp_name

        self.experiment = Experiment(
            env_id=env_id,
            eval=eval,
            seed=seed,
            tests=tests,
            track=track,
            grp_name=grp_name,
            run_name=run_name,
        )

        env_maps = {
            "hover_v0": HoverAviaryv0Config,
            "hover_v1": HoverAviaryv1Config,
            "traj_v2": TrajAviaryv2Config,
            "traj_v3": TrajAviaryv3Config,
            "traj_v2_ctbr": TrajAviaryv2CTBRConfig,
            "traj_v3_ctbr": TrajAviaryv3CTBRConfig,
            "traj_pay_v3": TrajAviaryPayv3Config,
        }

        try:
            try:
                self.environment = env_maps[env_id](
                    eval=eval, scale=scale, wind_bool=wind_bool
                )
            except TypeError:
                self.environment = env_maps[env_id](eval=eval, scale=scale)
            finally:
                if agent not in self.environment.agent_name:
                    raise ValueError("Provided agent does not match required agent")
        except KeyError:
            raise ValueError(f"Environment {env_id} not found in env_maps")

        self.scale = Scale(scale, scale_lengths=self.environment.scale_lengths)
        self.learning = Learning(seed=seed, test=tests, env_id=env_id)
        self.network = Network()

        for name, value in kwargs.items():
            if type(value) is Learning:
                self.learning = value
                self.learning.seed = seed
                self.learning.env_id = env_id

            if type(value) is Network:
                self.network = value


if __name__ == "__main__":
    cfg = Config()
