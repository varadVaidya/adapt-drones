from dataclasses import dataclass


@dataclass
class Experiment:
    env_id: str = "hover_v0"  # gym environment id
    eval: bool = False
    seed: int = 0  # random seed
    tests: bool = False  # testing mode
    track: bool = False  # track the experiment in wandb
    grp_name: str = "default"
    run_name: str = "default"
    wandb_project_name: str = "adapt-ICRA"
