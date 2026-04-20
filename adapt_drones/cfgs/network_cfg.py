from dataclasses import dataclass


@dataclass
class Network:
    base_policy_layers: [list, None] = None
    env_encoder_layers: [list, None] = None
    env_encoder_output: int = 4
    traj_encoder_output: int = 16
    adapt_time_horizon: int = 50

    def __post_init__(self):
        self.base_policy_layers = [16, 8]
        self.env_encoder_layers = [8]
