import os
import pkg_resources
import numpy as np
import mujoco
import gymnasium as gym
from gymnasium import spaces
from collections import OrderedDict
from typing import Union

from adapt_drones.envs.BaseAviaryPay import BaseAviaryPay
from adapt_drones.cfgs.config import Config
import adapt_drones.utils.rotation as rotation
import adapt_drones.utils.rewards as rewards
import adapt_drones.utils.visuals as visuals
from adapt_drones.utils.dynamics import CustomDynamics


class TrajAviaryPayv3(BaseAviaryPay):
    """
    Tracks the trajectory from slowed PITCN dataset.
    """

    # TODO: check is the action buffer is needed or not.
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
        super().__init__(
            init_xyz=init_xyz,
            init_quat=init_quat,
            mj_freq=mj_freq,
            ctrl_freq=ctrl_freq,
            record=record,
            camera_name=camera_name,
        )

        self.cfg: Config = cfg

        self.episode_length = cfg.environment.episode_length
        self.target_position = np.zeros(3)

        self.trajectory_dataset = cfg.environment.trajectory_dataset
        self.trajectory_window = cfg.environment.trajectory_window
        self.eval_trajectory_path = cfg.environment.eval_trajectory_path

        self.action_space = self._action_space()
        self.observation_space = self._observation_space()

        self.action_buffer = np.zeros((4, 4))

    def _action_space(self):
        lower_bound = -1 * np.ones(4)
        upper_bound = np.ones(4)

        return spaces.Box(low=lower_bound, high=upper_bound, dtype=np.float32)

    def _observation_space(self):
        """
        observation vector =
        [delta_pos, delta_ori, delta_vel, delta_angular_vel] = # 12
        For this environment, since the goal is to hover at a desired position,
        the desired velocity and angular velocity are zero, with [1,0,0,0] as the
        desired orientation.

        Adds the additional dynamics information to the observation space.
        this includes [mass, inertia, thrust2weight, prop_const, arm_length, wind] = [10]

        Adds the trajectory embeddings to the observation space.
        """

        lower_bound = lambda x: -1 * np.inf * np.ones(x)
        upper_bound = lambda x: np.inf * np.ones(x)

        state_box = spaces.Box(
            low=lower_bound(12), high=upper_bound(12), dtype=np.float32
        )
        self.state_obs_shape = state_box.shape[0]

        priv_shape = self.get_dynamics_info().shape
        self.priv_info_shape = priv_shape[0]

        priv_info_box = spaces.Box(
            low=lower_bound(priv_shape), high=upper_bound(priv_shape), dtype=np.float32
        )

        traj_window_shape = 6 * self.trajectory_window
        self.reference_traj_shape = traj_window_shape

        traj_box = spaces.Box(
            low=lower_bound(traj_window_shape),
            high=upper_bound(traj_window_shape),
            dtype=np.float32,
        )

        return spaces.Dict(
            {
                "priv_info": priv_info_box,
                "state": state_box,
                "trajectory": traj_box,
            }
        )

    def reset(self, seed=None, options=None):
        """Resets the environment, according to the gynmasium API."""

        super().reset(seed=seed, options=options)

        mujoco.mj_resetData(self.model, self.data)

        # trajectory reset
        self.reference_trajectory = self.reference_trajectory_reset()
        # kinematics reset
        self._kinematics_reset()
        # dynamics reset
        self._dynamics_reset(options=options)

        self.housekeeping()
        self.set_max_force_torque_limits()

        self.update_kinematic_data()

        initial_obs = self._compute_obs()
        initial_info = self._compute_info()

        return initial_obs, initial_info

    def step(self, action):
        """
        Extendeds the step method in the BaseAviary class to include action buffer
        """
        if self.cfg.environment.wind_bool:
            # change the wind slightly
            self.model.opt.wind += self.np_random.normal(0, 0.01, 3)
            self.model.opt.wind = np.clip(
                self.model.opt.wind,
                -self.cfg.environment.max_wind,
                self.cfg.environment.max_wind,
            )

        # add action noise to the action
        action += self.np_random.normal(0, 0.015, 4)
        action = np.clip(action, -1, 1)

        obs, reward, terminated, truncated, info = super().step(action)
        self.action_buffer = np.concatenate([self.action_buffer[1:], [action]])

        return obs, reward, truncated, False, info

    def _compute_obs(self):
        """
        observation vector =
        [delta_pos, delta_ori, delta_vel, delta_angular_vel] = # 12
        For this environment, since the goal is to hover at a desired position,
        the desired velocity and angular velocity are zero, with [1,0,0,0] as the
        desired orientation.
        """

        trajectory_window = self.get_trajectory_window()

        assert trajectory_window.shape == (self.trajectory_window, 6)
        window_position = trajectory_window[:, 0:3] - self.position
        window_velocity = trajectory_window[:, 3:6] - self.velocity

        self.target_position = trajectory_window[0, 0:3]
        self.target_velocity = trajectory_window[0, 3:6]

        delta_pos = self.target_position - self.position
        delta_vel = self.target_velocity - self.velocity

        ## add gaussian 0 mean noise to delta_pos and delta_vel
        delta_pos += self.np_random.normal(0, 0.02, 3)
        delta_vel += self.np_random.normal(0, 0.02, 3)

        delta_ori = np.zeros(3)
        quat = self.quat + self.np_random.normal(0, 0.01, 4)
        quat = quat / np.linalg.norm(quat)
        mujoco.mju_subQuat(delta_ori, quat, np.array([1.0, 0.0, 0.0, 0.0]))

        delta_angular_vel = np.zeros(3) - self.angular_velocity
        delta_angular_vel += self.np_random.normal(0, 0.001, 3)

        priv_info = self.get_dynamics_info()

        window_trajectory = (
            np.hstack([window_position, window_velocity]).flatten().astype(np.float32)
        )

        return dict(
            {
                "priv_info": priv_info,
                "state": np.hstack(
                    [delta_pos, delta_ori, delta_vel, delta_angular_vel]
                ).astype(np.float32),
                "trajectory": window_trajectory,
            }
        )

    def get_trajectory_window(self):
        """
        Gets the trajectory window for the current step.
        """

        start_idx = self.step_counter
        end_idx = start_idx + self.trajectory_window
        return self.reference_trajectory[start_idx:end_idx, 1:7]

    def reference_trajectory_reset(self):
        traj_index = self.np_random.integers(0, self.trajectory_dataset.shape[0])
        return self.trajectory_dataset[traj_index]

    def get_dynamics_info(self):
        """Returns the dynamics information of the drone
        used as privileged information for the network.
        """
        return np.hstack(
            [
                self.model.body_mass[self.drone_id],
                self.model.body_inertia[self.drone_id],
                self.thrust2weight,
                self.prop_const,
                self.arm_length,
                self.model.opt.wind,
            ]
        ).astype(np.float32)

    def _compute_reward(self):

        margin = 0.5
        isclose = 0.001
        norm_position = np.linalg.norm(self.target_position - self.position)
        norm_velocity = np.linalg.norm(self.target_velocity - self.velocity)
        norm_action = np.linalg.norm(np.diff(self.action_buffer, axis=0))

        distance_reward = rewards.tolerance(
            norm_position, bounds=(-isclose, isclose), margin=0.75
        )

        close_distance_reward = rewards.tolerance(
            norm_position, bounds=(-isclose, isclose), margin=0.075
        )

        velocity_reward = rewards.tolerance(
            norm_velocity, bounds=(-isclose, isclose), margin=margin
        )

        action_reward = rewards.tolerance(
            norm_action, bounds=(-isclose, isclose), margin=0.1
        )

        angular_velocity_reward = rewards.tolerance(
            np.linalg.norm(self.angular_velocity),
            bounds=(-isclose, isclose),
            margin=0.2,
        )

        weights = np.array([0.55, 0.15, 0.2, 0.05, 0.05])
        weights = weights / np.sum(weights)
        reward_vector = np.array(
            [
                distance_reward,
                close_distance_reward,
                velocity_reward,
                angular_velocity_reward,
                action_reward,
            ]
        )
        crash_reward = -100.0 if len(self.data.contact.dim) > 0 else 0.0

        reward = np.dot(weights, reward_vector) + crash_reward

        self.info_reward["distance_reward"] += distance_reward
        self.info_reward["velocity_reward"] += velocity_reward

        return np.float32(reward)

    def _compute_terminated(self):
        tolerance = 1e-8

        # explicit conversion to bool from numpy bool to satisfy the gym_checker
        # https://stackoverflow.com/questions/61791924/how-do-i-convert-an-array-of-numpy-booleans-to-python-booleans-for-serialization
        # answer from @Ch3steR
        # TODO: LOW: check for a better way to convert numpy bool to python bool
        return (
            (np.linalg.norm(self.position - self.target_position) < tolerance)
            .astype(bool)
            .tolist()
        )

    def _compute_truncated(self):
        far_away = np.linalg.norm(self.position) > 7.5
        crashed = len(self.data.contact.dim) > 0

        return far_away or crashed

    def _compute_info(self):
        self.info_pos_error += np.linalg.norm(self.target_position - self.position)
        self.info_vel_error += np.linalg.norm(self.velocity)

        error_dict = {
            "pos_error": self.info_pos_error,
            "vel_error": self.info_vel_error,
        }
        error_dict.update(self.info_reward)

        return error_dict

    def _kinematics_reset(self):
        """
        Resets the kinematics of the drone.
        """

        _pos_xy = self.cfg.environment.pos_xy
        _pos_z = self.cfg.environment.pos_z

        init_pos = np.zeros(7)
        delta_pos = np.hstack(
            (
                self.np_random.uniform(_pos_xy[0], _pos_xy[1], 2),
                self.np_random.uniform(_pos_z[0], _pos_z[1], 1),
            )
        )
        init_pos[0:3] = self.reference_trajectory[0][1:4] + delta_pos

        _roll_pitch = self.cfg.environment.roll_pitch

        euler = np.concatenate(
            [self.np_random.uniform(_roll_pitch[0], _roll_pitch[1], 2), [0.0]]
        )
        mat = rotation.euler2mat(euler).reshape(9)
        mujoco.mju_mat2Quat(init_pos[3:7], mat)

        self.data.qpos = np.hstack(
            (
                init_pos,
                init_pos[0:3] - np.array([0.0, 0.0, 0.25]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        _lin_vel = self.cfg.environment.linear_vel
        _ang_vel = self.cfg.environment.angular_vel

        delta_vel = self.np_random.uniform(_lin_vel[0], _lin_vel[1], 3)

        self.data.qvel = np.concatenate(
            [
                self.reference_trajectory[0][4:7] + delta_vel,
                self.np_random.uniform(_ang_vel[0], _ang_vel[1], 3),
                np.zeros(6),
            ]
        )

    def _dynamics_reset(self, options: [Union[None, dict]] = None):
        """
        Resets the dynamics properties of the drone. Uses the scaling laws to
        reset the properties.
        """

        if options is None:
            # arm length
            _arm_length = self.cfg.scale.scale_lengths
            self.arm_length = self.np_random.uniform(_arm_length[0], _arm_length[1])
            L = self.arm_length

            # mass
            _mass_avg = np.polyval(self.cfg.scale.avg_mass_fit, L)
            _mass_std = np.polyval(self.cfg.scale.std_mass_fit, L)
            _mass_std = 0.0 if _mass_std < 0.0 else _mass_std

            _mass = self.np_random.uniform(_mass_avg - _mass_std, _mass_avg + _mass_std)
            self.model.body_mass[self.drone_id] = _mass

            self.HOVER_THRUST = _mass * -1 * self.model.opt.gravity[2]

            # ixx
            _ixx_avg = np.polyval(self.cfg.scale.avg_ixx_fit, L)
            _ixx_std = np.polyval(self.cfg.scale.std_ixx_fit, L)
            _ixx_std = 0.0 if _ixx_std < 0.0 else _ixx_std

            _ixx = self.np_random.uniform(_ixx_avg - _ixx_std, _ixx_avg + _ixx_std)

            # iyy
            _iyy_avg = np.polyval(self.cfg.scale.avg_iyy_fit, L)
            _iyy_std = np.polyval(self.cfg.scale.std_iyy_fit, L)
            _iyy_std = 0.0 if _iyy_std < 0.0 else _iyy_std

            _iyy = self.np_random.uniform(_iyy_avg - _iyy_std, _iyy_avg + _iyy_std)

            # izz
            _izz_avg = np.polyval(self.cfg.scale.avg_izz_fit, L)
            _izz_std = np.polyval(self.cfg.scale.std_izz_fit, L)
            _izz_std = 0.0 if _izz_std < 0.0 else _izz_std

            _izz = self.np_random.uniform(_izz_avg - _izz_std, _izz_avg + _izz_std)

            inertia = np.array([_ixx, _iyy, _izz]).reshape(3)
            self.model.body_inertia[self.drone_id] = inertia

            # com offset 5% of the arm length in xy and 2.5% in z
            com_offset = 0.05 * L
            com_xy = self.np_random.uniform(-com_offset, com_offset, 2)
            com_offset = 0.025 * L
            com_z = self.np_random.uniform(-com_offset, com_offset, 1)

            com = np.hstack((com_xy, com_z))
            self.model.body_ipos = com  # drone com

            # thrust offset 5% of the arm length in xy and 2.5% in z
            thrust_offset = 0.05 * L
            thrust_xy = self.np_random.uniform(-thrust_offset, thrust_offset, 2)
            thrust_offset = 0.025 * L
            thrust_z = self.np_random.uniform(-thrust_offset, thrust_offset, 1)

            thrust = np.hstack((thrust_xy, thrust_z))
            self.model.site_pos[self.com_site_id] = thrust  # thrust site

            # km_kf
            _km_kf_avg = np.abs(
                np.polyval(self.cfg.scale.avg_km_kf_fit, self.arm_length)
            )
            _km_kf_std = np.polyval(self.cfg.scale.std_km_kf_fit, self.arm_length)
            _km_kf_std = 0.0 if _km_kf_std < 0.0 else _km_kf_std

            while _km_kf_avg - _km_kf_std < 5e-4:
                _km_kf_std *= 0.9

            _km_kf = self.np_random.uniform(
                _km_kf_avg - _km_kf_std, _km_kf_avg + _km_kf_std
            )
            self.prop_const = _km_kf

        else:
            dynamics = CustomDynamics(**options)
            L = dynamics.arm_length

            self.model.body_mass[self.drone_id] = dynamics.mass
            self.HOVER_THRUST = dynamics.mass * -1 * self.model.opt.gravity[2]

            inertia = np.array([dynamics.ixx, dynamics.iyy, dynamics.izz]).reshape(3)
            self.model.body_inertia[self.drone_id] = inertia

            com_offset = 0.05 * L
            com_xy = self.np_random.uniform(-com_offset, com_offset, 2)
            com_offset = 0.025 * L
            com_z = self.np_random.uniform(-com_offset, com_offset, 1)

            com = np.hstack((com_xy, com_z))
            self.model.body_ipos = com  # drone com

            thrust_offset = 0.05 * L
            thrust_xy = self.np_random.uniform(-thrust_offset, thrust_offset, 2)
            thrust_offset = 0.025 * L
            thrust_z = self.np_random.uniform(-thrust_offset, thrust_offset, 1)

            thrust = np.hstack((thrust_xy, thrust_z))
            self.model.site_pos[self.com_site_id] = thrust

            self.prop_const = dynamics.km_kf

        self.thrust2weight = 2.75

        mujoco.mj_setConst(self.model, self.data)

        # TODO: add wind.
        wind_magnitude = self.np_random.uniform(
            self.cfg.environment.wind_speed[0], self.cfg.environment.wind_speed[1]
        )
        wind_direction = self.np_random.uniform(-1, 1, 3)
        wind_direction = wind_direction / np.linalg.norm(wind_direction)
        self.model.opt.wind = wind_magnitude * wind_direction

    def eval_trajectory(self, idx=None, trajectory=None):
        """Evaluation method that will be called by the eval script to
        generate the trajectory for the evaluation.
        Assumes that eval is set by the config file

        Args:
        idx: int: The index of the trajectory to be evaluated. If None, a trajectory is
        sampled randomly from the evaluation dataset.
        """
        if trajectory is None:
            eval_trajs = np.load(self.eval_trajectory_path)

            idx = (
                self.np_random.integers(0, eval_trajs.shape[0]) if idx is None else idx
            )
            eval_traj = eval_trajs[idx]
            # print("eval traj", eval_traj.shape)
            rows_not_nan = sum(~np.isnan(eval_traj[:, 1]))
            self.reference_trajectory = eval_traj[:rows_not_nan]

            self._kinematics_reset()
            duration = self.reference_trajectory.shape[0] - (self.trajectory_window + 1)

            t = np.linspace(0, duration / self.mj_freq, duration)

            reference_position = eval_traj[:duration, 1:4]
            reference_velocity = eval_traj[:duration, 4:7]

            return t, reference_position, reference_velocity

        else:
            t, pos, vel = trajectory
            self.reference_trajectory = np.concatenate([t.reshape(-1, 1), pos, vel], 1)

            self.data.qpos = np.concatenate([pos[0], np.array([1, 0, 0, 0])])
            self.data.qvel = np.concatenate([vel[0], np.zeros(3)])

            duration = self.reference_trajectory.shape[0] - (self.trajectory_window + 1)
            t = np.linspace(0, duration / self.mj_freq, duration)

            reference_position = pos[:duration, :]
            reference_velocity = vel[:duration, :]

            return t, reference_position, reference_velocity

    def housekeeping(self):
        """
        Adds more functionality to the housekeeping method in the BaseAviary class.
        """
        super().housekeeping()
        self.info_pos_error = 0.0
        self.info_vel_error = 0.0
        self.info_reward = {
            "distance_reward": 0.0,
            "velocity_reward": 0.0,
        }

    def init_render(self):
        """
        Adds refrence trace positions
        """
        import collections

        super().init_render()
        self._trace_ref_positions = collections.deque(maxlen=50)
        self._wind_arrow = -1
        self._wind_force_arrow = -1

    def render_frame(self):
        """
        Renders the frame for the environment.
        """

        if len(self._frames) < self.data.time * self.recording_FPS:
            self.renderer.update_scene(self.data, self.camera_name)
            self._trace_positions.append(self.position)
            self._trace_ref_positions.append(self.target_position)
            visuals.modify_scene(
                self.renderer.scene, self._trace_positions, self._trace_ref_positions
            )
            # self._wind_arrow = visuals.render_vector(
            #     scene=self.renderer.scene,
            #     vector=self.model.opt.wind,
            #     pos=self.position + np.array([0, 0, 0.05]),
            #     scale=0.5,
            #     color=np.array([0, 1, 0, 1]),
            #     # geom_id=self._wind_arrow,
            # )

            # self._wind_force_arrow = visuals.render_vector(
            #     scene=self.renderer.scene,
            #     vector=self.data.qfrc_passive[:3],
            #     pos=self.position + np.array([0, 0, 0.05]),
            #     scale=0.5,
            #     color=np.array([0, 0, 1, 1]),
            #     # geom_id=self._wind_force_arrow,
            # )

            frame = self.renderer.render()
            self._frames.append(frame)


if __name__ == "__main__":
    cfg = Config()
    env = HoverAviaryv0(cfg)
    env.reset()
    env.eval_trajectory(10)
    # env.step(env.action_space.sample())
    # env.close()
