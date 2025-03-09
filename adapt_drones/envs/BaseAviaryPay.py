import os
import time
import collections
import pkg_resources
from typing import List

import numpy as np
import mujoco
import mujoco.viewer
import gymnasium as gym
import ffmpeg

import adapt_drones.utils.visuals as visuals

# TODO: think if action buffer is needed


class BaseAviaryPay(gym.Env):
    "Base class for all environments"

    metadata = {"render_modes": ["human"], "render_fps": 4}

    def __init__(
        self,
        init_xyz=None,
        init_quat=None,
        mj_freq: int = 100,
        ctrl_freq: int = 100,
        record: bool = False,
        camera_name: str = "trackcom",
    ):
        """
        Initialize the environment
        """

        self.mj_freq = mj_freq
        self.ctrl_freq = ctrl_freq

        if self.mj_freq % self.ctrl_freq != 0:
            raise ValueError(
                "Control frequency must be a factor of the simulation frequency"
            )

        self.mj_steps_per_ctrl = int(self.mj_freq / self.ctrl_freq)
        self.ctrl_timestep = 1.0 / self.ctrl_freq
        self.mj_timestep = 1.0 / self.mj_freq

        # Options
        self.record = record
        self.recording_height = 480
        self.recording_width = 720
        self.recording_FPS = 25
        self.camera_name = camera_name

        # MuJoCo modeladapt_drones
        xml_file = pkg_resources.resource_filename(
            "adapt_drones", "assets/quad_payload.xml"
        )

        if not os.path.exists(xml_file):
            raise FileNotFoundError(f"File {xml_file} not found")

        self.model: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_file)
        self.data: mujoco.MjData = mujoco.MjData(self.model)

        if init_xyz is None:
            init_xyz = np.array([0.0, 0.0, 1.0])
        if init_quat is None:
            init_quat = np.array([1.0, 0.0, 0.0, 0.0])

        self.init_xyz = np.array(init_xyz).copy()
        self.init_quat = np.array(init_quat).copy()
        self.init_pose = np.concatenate([self.init_xyz, self.init_quat])

        self.data.qpos = self.data.qpos = np.hstack(
            (
                self.init_pose,
                self.init_pose[0:3] - np.array([0.0, 0.0, 0.25]),
                np.array([1.0, 0.0, 0.0, 0.0]),
            )
        )

        self.model.opt.timestep = self.mj_timestep
        self.drone_id = self.data.body("quad").id
        self.com_site_id = mujoco.mj_name2id(self.model, 6, "thrust_com")

        self.thrust2weight = 3.5
        self.prop_const = 0.014
        self.arm_length = 0.166

        self.HOVER_THRUST = (
            self.model.body_mass[self.drone_id] * -1 * self.model.opt.gravity[2]
        )

        self.set_max_force_torque_limits()

        # Housekeeping
        self.housekeeping()
        self.render_mode = "human"
        self.update_kinematic_data()

        # recording
        if self.record:
            self.init_render()

    def step(self, action: np.ndarray):
        """Take a step in the environment

        Args:
            action (np.ndarray): Action to be taken

        Returns:
            Tuple[np.ndarray, float, bool, bool, dict]: State, reward, terminated, truncated, info

        """
        force_torque_action = self.preprocess_action(action)

        for _ in range(self.mj_steps_per_ctrl):
            # self.update_kinematic_data() # TODO: check if this is needed
            self.data.ctrl = force_torque_action
            mujoco.mj_step(self.model, self.data)

            if self.record:
                self.render_frame()

        self.last_force_torque_action = force_torque_action
        self.update_kinematic_data()

        obs = self._compute_obs()
        terminated = self._compute_terminated()
        truncated = self._compute_truncated()
        reward = self._compute_reward()
        info = self._compute_info()

        self.step_counter += 1

        return obs, reward, terminated, truncated, info

    def preprocess_action(self, action: np.ndarray):
        """Pre-processes the action passed to `.step()` before it is applied to the environment.
        Converts the normalised action into body thrust and torques.
        i.e. converts the
        action[0] from [-1,1] to [0,MAX_THRUST] such that 0 action corresponds to hover thrust
        action[1:] from [-1,1] to [-MAX_TORQUE,MAX_TORQUE]
        """
        action = np.clip(action, -1, 1)

        force_torque_action = np.zeros(4)

        multipler = (self.thrust2weight - 1.0) if action[0] > 0.0 else 1.0
        normalized_thrust = action[0] * multipler + 1.0

        force_torque_action[0] = normalized_thrust * self.HOVER_THRUST
        force_torque_action[1:] = action[1:] * self.max_torque

        return force_torque_action.astype(np.float32)

    def housekeeping(self):
        """Housekeeping function that keeps everrything organised.
        To be called after every reset.

        Child classes to super this method and add their own housekeeping
        if needed.
        """

        self.reset_time = time.time()
        self.step_counter = 0

        self.last_force_torque_action = np.zeros(4)

        self.position = np.zeros((3, 1))
        self.quat = np.zeros((4, 1))
        self.velocity = np.zeros((3, 1))
        self.angular_velocity = np.zeros((3, 1))

        self.model.opt.gravity[2] = -9.81
        self.model.opt.timestep = self.mj_timestep
        self.HOVER_THRUST = (
            self.model.body_mass[self.drone_id] * -1 * self.model.opt.gravity[2]
        )

    def set_max_force_torque_limits(self):

        max_thrust = self.HOVER_THRUST * self.thrust2weight
        single_motor_max_thrust = max_thrust / 4
        max_torque_xy = single_motor_max_thrust * self.arm_length
        max_torque_z = 2 * single_motor_max_thrust * self.prop_const

        self.max_thrust = max_thrust
        self.max_torque = np.array([max_torque_xy, max_torque_xy, max_torque_z])

    def update_kinematic_data(self):
        """Update the kinematic data of the drone"""

        self.position = self.data.qpos[0:3].copy()
        self.quat = self.data.qpos[3:7].copy()
        self.velocity = self.data.qvel[0:3].copy()
        self.angular_velocity = self.data.qvel[3:6].copy()

    def get_drone_state_info(self, return_action: bool = False):
        """Get the state information of the drone

        Returns:
            np.ndarray: State information of the drone
        """
        state = np.concatenate(
            [
                self.position,
                self.quat,
                self.velocity,
                self.angular_velocity,
            ]
        )

        if return_action:
            state = np.concatenate([state, self.last_force_torque_action])

        return state

    def _compute_obs(self):
        """Compute the observation of the environment
        Must be implemented by the child classes
        """
        raise NotImplementedError

    def _compute_terminated(self):
        """
        Compute if the episode is terminated
        Must be implemented by the child classes
        """

        raise NotImplementedError

    def _compute_truncated(self):
        """
        Compute if the episode is truncated
        Must be implemented by the child classes
        """
        raise NotImplementedError

    def _compute_reward(self):
        """
        Compute the reward of the environment
        Must be implemented by the child classes
        """
        raise NotImplementedError

    def _compute_info(self):
        """
        Compute the info of the environment
        """
        raise NotImplementedError

    def init_render(self):
        """initialize the renderer
        Child classes to add their own rendering methods
        using super to modify the rendering
        """
        self.renderer = mujoco.Renderer(
            self.model, height=self.recording_height, width=self.recording_width
        )
        self._frames = []
        self._trace_positions = collections.deque(maxlen=50)
        self.renderer.update_scene(self.data, self.camera_name)

        frame = self.renderer.render()
        self._frames.append(frame)

    def render_frame(self):

        if len(self._frames) < self.data.time * self.recording_FPS:
            self.renderer.update_scene(self.data, self.camera_name)
            self._trace_positions.append(self.position)
            visuals.modify_scene(self.renderer.scene, self._trace_positions)
            frame = self.renderer.render()
            self._frames.append(frame)

    def vidwrite(self, fn, vcodec="libx264"):
        framerate = self.recording_FPS
        images = self._frames
        if not isinstance(images, np.ndarray):
            images = np.asarray(images)
        n, height, width, channels = images.shape
        process = (
            ffmpeg.input(
                "pipe:",
                format="rawvideo",
                pix_fmt="rgb24",
                r=framerate,
                s="{}x{}".format(width, height),
            )
            .output(fn, pix_fmt="yuv420p", vcodec=vcodec, loglevel="quiet")
            .overwrite_output()
            .run_async(pipe_stdin=True)
        )
        for frame in images:
            process.stdin.write(frame.astype(np.uint8).tobytes())
        process.stdin.close()
        process.wait()


if __name__ == "__main__":
    env = BaseAviary()
