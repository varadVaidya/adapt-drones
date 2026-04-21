import pkg_resources
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces

from adapt_drones.envs.TrajAviaryv3 import TrajAviaryv3
from adapt_drones.cfgs.config import Config


class TrajObstacleAviaryv3(TrajAviaryv3):

    def __init__(
        self,
        cfg: Config,
        init_xyz=None,
        init_quat=None,
        mj_freq: int = 100,
        ctrl_freq: int = 100,
        record: bool = False,
        camera_name: str = "trackcom",
    ):
        self.n_obstacles = cfg.environment.n_obstacles
        self.obstacle_obs_shape = self.n_obstacles * 3
        self.obstacle_radius_range = cfg.environment.obstacle_radius_range
        self.danger_radius = cfg.environment.danger_radius

        xml_file = pkg_resources.resource_filename(
            "adapt_drones", "assets/quad_obstacles.xml"
        )
        super().__init__(
            cfg=cfg,
            init_xyz=init_xyz,
            init_quat=init_quat,
            mj_freq=mj_freq,
            ctrl_freq=ctrl_freq,
            record=record,
            camera_name=camera_name,
            xml_file=xml_file,
        )

        self.obstacle_body_ids = [
            self.data.body(f"obstacle_{i}").id for i in range(self.n_obstacles)
        ]
        self.obstacle_geom_ids = [
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, f"obs_geom_{i}")
            for i in range(self.n_obstacles)
        ]

        self.drone_geom_names = [
            "quad_geom",
            "arm_front_left",
            "arm_front_right",
            "arm_back_right",
            "arm_back_left",
            "thruster_front_left",
            "thruster_front_right",
            "thruster_back_right",
            "thruster_back_left",
        ]
        self.drone_geom_ids = set(
            mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
            for name in self.drone_geom_names
        )
        self.obstacle_geom_id_set = set(self.obstacle_geom_ids)

    def _observation_space(self):
        obs_space = super()._observation_space()

        state_dim = obs_space.spaces["state"].shape[0]
        new_state_dim = state_dim + self.obstacle_obs_shape
        obs_space.spaces["state"] = spaces.Box(
            low=-np.inf * np.ones(new_state_dim),
            high=np.inf * np.ones(new_state_dim),
            dtype=np.float32,
        )
        self.state_obs_shape = new_state_dim

        return obs_space

    def _compute_obs(self):
        obs = super()._compute_obs()
        obstacle_obs = self._get_obstacle_obs()
        obs["state"] = np.concatenate([obs["state"], obstacle_obs])
        return obs

    def _get_obstacle_obs(self):
        rel_positions = []
        for body_id in self.obstacle_body_ids:
            obs_pos = self.data.xpos[body_id]
            rel_pos = obs_pos - self.position
            rel_positions.append(rel_pos)
        rel_positions = sorted(rel_positions, key=lambda p: np.linalg.norm(p))
        return np.concatenate(rel_positions).astype(np.float32)

    def reset(self, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        self._randomize_obstacles()
        mujoco.mj_forward(self.model, self.data)
        self.update_kinematic_data()

        obs = self._compute_obs()
        info = self._compute_info()
        return obs, info

    def _randomize_obstacles(self):
        traj_positions = self.reference_trajectory[:, 1:4]
        traj_len = len(traj_positions)
        margin = int(max(traj_len // 3, 50))

        for i in range(self.n_obstacles):
            t_idx = self.np_random.integers(margin, traj_len - margin)
            base_pos = traj_positions[t_idx].copy()

            # Compute local trajectory direction at this point
            t_next = min(t_idx + 5, traj_len - 1)
            t_prev = max(t_idx - 5, 0)
            traj_dir = traj_positions[t_next] - traj_positions[t_prev]
            traj_dir_norm = np.linalg.norm(traj_dir)
            if traj_dir_norm > 1e-6:
                traj_dir = traj_dir / traj_dir_norm
            else:
                traj_dir = np.array([1.0, 0.0, 0.0])

            # Build perpendicular basis
            up = np.array([0.0, 0.0, 1.0])
            perp1 = np.cross(traj_dir, up)
            if np.linalg.norm(perp1) < 1e-6:
                up = np.array([0.0, 1.0, 0.0])
                perp1 = np.cross(traj_dir, up)
            perp1 = perp1 / np.linalg.norm(perp1)
            perp2 = np.cross(traj_dir, perp1)

            # Place obstacle perpendicular to trajectory, close enough to matter
            # Perpendicular offset: 0.05-0.25m (close to or on the trajectory)
            # Along-trajectory jitter: small, to avoid stacking on one point
            perp_dist = self.np_random.uniform(0.05, 0.25)
            perp_angle = self.np_random.uniform(0, 2 * np.pi)
            along_jitter = self.np_random.uniform(-0.1, 0.1)

            offset = (
                perp_dist * (np.cos(perp_angle) * perp1 + np.sin(perp_angle) * perp2)
                + along_jitter * traj_dir
            )
            self.model.body_pos[self.obstacle_body_ids[i]] = base_pos + offset

            radius = self.np_random.uniform(*self.obstacle_radius_range)
            self.model.geom_size[self.obstacle_geom_ids[i]][0] = radius

    def _min_obstacle_distance(self):
        min_dist = np.inf
        for i, body_id in enumerate(self.obstacle_body_ids):
            obs_pos = self.data.xpos[body_id]
            radius = self.model.geom_size[self.obstacle_geom_ids[i]][0]
            dist = np.linalg.norm(obs_pos - self.position) - radius
            min_dist = min(min_dist, dist)
        return min_dist

    def _check_obstacle_collision(self):
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            drone_hit = geom1 in self.drone_geom_ids or geom2 in self.drone_geom_ids
            obs_hit = (
                geom1 in self.obstacle_geom_id_set or geom2 in self.obstacle_geom_id_set
            )
            if drone_hit and obs_hit:
                return True
        return False

    def _compute_truncated(self):
        far_away = np.linalg.norm(self.position) > 7.5
        obstacle_collision = self._check_obstacle_collision()

        ground_crash = False
        for i in range(self.data.ncon):
            contact = self.data.contact[i]
            geom1, geom2 = contact.geom1, contact.geom2
            is_drone = geom1 in self.drone_geom_ids or geom2 in self.drone_geom_ids
            is_obstacle = (
                geom1 in self.obstacle_geom_id_set or geom2 in self.obstacle_geom_id_set
            )
            if is_drone and not is_obstacle:
                ground_crash = True
                break

        return far_away or obstacle_collision or ground_crash

    def _compute_info(self):
        info = super()._compute_info()
        min_dist = self._min_obstacle_distance()
        info["cost"] = np.float32(1.0 if min_dist < self.danger_radius else 0.0)
        info["min_obstacle_distance"] = np.float32(min_dist)
        info["obstacle_collision"] = np.float32(self._check_obstacle_collision())
        return info
