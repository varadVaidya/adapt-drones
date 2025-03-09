from dataclasses import dataclass


@dataclass
class Network:
    base_policy_layers: [list, None] = None
    env_encoder_layers: [list, None] = None
    env_encoder_output: int = 8
    traj_encoder_output: int = 32
    adapt_time_horizon: int = 50

    def __post_init__(self):
        self.base_policy_layers = [64, 64, 64]
        self.env_encoder_layers = [64, 64]
