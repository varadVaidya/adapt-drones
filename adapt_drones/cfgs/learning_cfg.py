from dataclasses import dataclass


@dataclass
class Learning:
    env_id: str = "hover_v0"
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    test: bool = False
    total_timesteps: int = 60000000
    init_lr: float = 2.5e-4
    num_envs: int = 64
    num_steps: int = 1024
    anneal_lr: bool = True
    final_lr: float = 2.24e-4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    num_minibatches: int = 64
    save_model: bool = True

    update_epochs: int = 10
    norm_adv: bool = True
    clip_coef: float = 0.2
    clip_vloss: bool = False
    ent_coef: float = 0.0005
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    target_kl: None = None

    # to be filled in post_init
    batch_size: int = 0
    minibatch_size: int = 0
    num_iterations: int = 0

    def __post_init__(self):
        if self.test:
            self.total_timesteps = 15000
            self.num_envs = 4
            self.num_steps = 512
        self.batch_size = int(self.num_envs * self.num_steps)
        self.minibatch_size = int(self.batch_size // self.num_minibatches)
        self.num_iterations = self.total_timesteps // self.batch_size
