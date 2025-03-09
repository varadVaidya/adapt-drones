from dataclasses import dataclass
import pkg_resources
import numpy as np


@dataclass
class HoverAviaryv0Config:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    target_pos_xy: list
    target_pos_z: list

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    env_id: str = "hover_v0"
    episode_length: int = 6  # secs
    agent_name: str = "AC"

    def __init__(self, eval, scale):
        self.eval = eval
        self.scale = scale

        self.pos_xy = [-1.0, 1.0] if not eval else [-1.25, 1.25]
        self.pos_z = [1.0, 2.0] if not eval else [0.75, 2.25]

        self.target_pos_xy = [-1.0, 1.0] if not eval else [-1.25, 1.25]
        self.target_pos_z = [1.0, 2.0] if not eval else [0.75, 2.25]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.16, 0.16]


@dataclass
class HoverAviaryv1Config:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    target_pos_xy: list
    target_pos_z: list

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    env_id: str = "hover_v1"
    episode_length: int = 6  # secs
    agent_name: tuple = ("AC", "RMA")

    def __init__(self, eval, scale):
        self.eval = eval
        self.scale = scale

        self.pos_xy = [-1.0, 1.0] if not eval else [-1.25, 1.25]
        self.pos_z = [1.0, 2.0] if not eval else [0.75, 2.25]

        self.target_pos_xy = [-1.0, 1.0] if not eval else [-1.25, 1.25]
        self.target_pos_z = [1.0, 2.0] if not eval else [0.75, 2.25]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.16, 0.16]


@dataclass
class TrajAviaryv2Config:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    env_id: str = "traj_v2"
    episode_length: int = 6  # secs
    agent_name: tuple = ("RMA_DATT",)
    trajectory_window: int = 100

    def __init__(self, eval, scale):
        self.eval = eval
        self.scale = scale

        # pos_xy here refers to the delta in the drone's position
        # compared the reference position
        self.pos_xy = [-0.10, 0.10] if not eval else [-0.10, 0.10]
        self.pos_z = [-0.1, 0.1] if not eval else [-0.1, 0.1]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.05, 0.05]

        trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/crazy_trajs_train.npy"
        )
        self.trajectory_dataset = np.load(trajectory_path)
        self.eval_trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/crazy_trajs_eval.npy"
        )


@dataclass
class TrajAviaryv3Config:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    wind_speed: list
    max_wind: float

    env_id: str = "traj_v3"
    episode_length: int = 6  # secs
    agent_name: tuple = ("RMA_DATT",)
    wind_bool: bool = True
    trajectory_window: int = 100

    def __init__(self, eval, scale, wind_bool):
        self.eval = eval
        self.scale = scale

        # pos_xy here refers to the delta in the drone's position
        # compared the reference position
        self.pos_xy = [-0.10, 0.10] if not eval else [-0.10, 0.10]
        self.pos_z = [-0.1, 0.1] if not eval else [-0.1, 0.1]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.05, 0.05]

        self.wind_bool = wind_bool
        self.wind_speed = [0.0, 1.5] if not eval else [0.0, 1.75]
        self.max_wind = 2.0 if not eval else 2.0

        self.wind_speed = self.wind_speed if self.wind_bool else [0.0, 0.0]
        self.max_wind = self.max_wind if self.wind_bool else 0.0

        trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/slow_pi_tcn_train.npy"
        )
        self.trajectory_dataset = np.load(trajectory_path, allow_pickle=True)
        self.eval_trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/slow_pi_tcn_eval.npy"
        )


@dataclass
class TrajAviaryPayv3Config:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    wind_speed: list
    max_wind: float

    env_id: str = "traj_v3"
    episode_length: int = 6  # secs
    agent_name: tuple = ("RMA_DATT",)
    wind_bool: bool = True
    trajectory_window: int = 100

    def __init__(self, eval, scale, wind_bool):
        self.eval = eval
        self.scale = scale

        # pos_xy here refers to the delta in the drone's position
        # compared the reference position
        self.pos_xy = [-0.10, 0.10] if not eval else [-0.10, 0.10]
        self.pos_z = [-0.1, 0.1] if not eval else [-0.1, 0.1]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.05, 0.05]

        self.wind_bool = wind_bool
        self.wind_speed = [0.0, 1.5] if not eval else [0.0, 1.75]
        self.max_wind = 2.0 if not eval else 2.0

        self.wind_speed = self.wind_speed if self.wind_bool else [0.0, 0.0]
        self.max_wind = self.max_wind if self.wind_bool else 0.0

        trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/slow_pi_tcn_train.npy"
        )
        self.trajectory_dataset = np.load(trajectory_path, allow_pickle=True)
        self.eval_trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/slow_pi_tcn_eval.npy"
        )


@dataclass
class TrajAviaryv2CTBRConfig:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    env_id: str = "traj_v2"
    episode_length: int = 6  # secs
    agent_name: tuple = ("RMA_DATT",)
    trajectory_window: int = 50

    def __init__(self, eval, scale):
        self.eval = eval
        self.scale = scale

        # pos_xy here refers to the delta in the drone's position
        # compared the reference position
        self.pos_xy = [-0.25, 0.25] if not eval else [-0.25, 0.25]
        self.pos_z = [-0.25, 0.25] if not eval else [-0.25, 0.25]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.16, 0.16]

        trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/crazy_trajs_train.npy"
        )
        self.trajectory_dataset = np.load(trajectory_path)
        self.eval_trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/crazy_trajs_eval.npy"
        )


@dataclass
class TrajAviaryv3CTBRConfig:
    eval: bool

    scale: bool
    scale_lengths: list

    pos_xy: list
    pos_z: float

    linear_vel: list
    angular_vel: list

    roll_pitch: list

    env_id: str = "traj_v3_ctbr"
    episode_length: int = 6  # secs
    agent_name: tuple = ("RMA_DATT",)
    trajectory_window: int = 100

    def __init__(self, eval, scale):
        self.eval = eval
        self.scale = scale

        # pos_xy here refers to the delta in the drone's position
        # compared the reference position
        self.pos_xy = [-0.10, 0.10] if not eval else [-0.10, 0.10]
        self.pos_z = [-0.1, 0.1] if not eval else [-0.1, 0.1]

        self.linear_vel = [-0.1, 0.1] if not eval else [-0.125, 0.125]
        self.angular_vel = [-0.05, 0.05] if not eval else [-0.05, 0.05]

        self.roll_pitch = [-0.15, 0.15] if not eval else [-0.15, 0.15]

        self.scale_lengths = [0.05, 0.16] if self.scale else [0.05, 0.05]

        trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/slow_pi_tcn_train.npy"
        )
        self.trajectory_dataset = np.load(trajectory_path)
        self.eval_trajectory_path = pkg_resources.resource_filename(
            "adapt_drones", "assets/slow_pi_tcn_eval.npy"
        )
