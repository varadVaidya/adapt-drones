import numpy as np

## quaternion utils for rotation. NOTE: Keeping all of the functions in the file different from the original code
## This tbh doesn't matter much, but i want to keep the emulation code independent of the original code


def normalise_quat(q):
    return q / np.linalg.norm(q)


def conj_quat(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def vec_to_quat(v):
    """Converts a 3D vector to a quaternion."""
    q = np.zeros(4)
    q[1:] = v
    return q


def q_mult(q1, q2):
    a = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    b = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    c = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    d = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

    return np.array([a, b, c, d])


def rotate(q, v):
    """Rotate vector v by quaternion q."""
    q = normalise_quat(q)
    v = vec_to_quat(v)
    return q_mult(q, q_mult(v, conj_quat(q)))[1:]


def q_exp(q):
    """Exponential map of a quaternion."""
    a = q[0]
    v = q[1:]

    v_norm = np.linalg.norm(v)
    e_a = np.exp(a)

    exp_q = np.zeros(4)
    exp_q[0] = e_a * np.cos(v_norm)

    if v_norm > 1e-6:
        exp_q[1:] = e_a * np.sin(v_norm) * v / v_norm

    return exp_q


def sub_quat(qa, qb):
    qb = conj_quat(qb)
    qdiff = q_mult(qb, qa)

    return quat_to_vel(qdiff, 1.0)


def quat_to_vel(quat, dt):

    axis = quat[1:]

    sin_a_2 = np.linalg.norm(axis)
    if sin_a_2 > 0:
        axis /= sin_a_2

    angle = 2 * np.arctan2(sin_a_2, quat[0])

    angle = angle if angle <= np.pi else angle - 2 * np.pi

    return axis * angle / dt


def skew_symmetric(v):

    return np.array(
        [
            [0, -v[0], -v[1], -v[2]],
            [v[0], 0, v[2], -v[1]],
            [v[1], -v[2], 0, v[0]],
            [v[2], v[1], -v[0], 0],
        ]
    )


def q_integrate(q, v, dt):
    """
    Integrate a quaternion for a timestep dt given angular velocity v
    """
    return q_mult(q_exp(vec_to_quat(v * dt / 2)), q)


def q_integrate2(q, v, dt):
    """
    Integrate a quaternion for a timestep dt given angular velocity v
    """
    return q_mult(q_exp(vec_to_quat(v * dt / 2)), q)


class State:
    """Class that stores the state of a UAV as used in the simulator interface."""

    def __init__(
        self,
        pos=np.zeros(3),
        vel=np.zeros(3),
        quat=np.array([1, 0, 0, 0]),
        omega=np.zeros(3),
    ):
        # internally use one numpy array
        self._state = np.empty(13)
        self.pos = pos
        self.vel = vel
        self.quat = quat
        self.omega = omega

    @property
    def pos(self):
        """Position [m; world frame]."""
        return self._state[0:3]

    @pos.setter
    def pos(self, value):
        self._state[0:3] = value

    @property
    def vel(self):
        """Velocity [m/s; world frame]."""
        return self._state[3:6]

    @vel.setter
    def vel(self, value):
        self._state[3:6] = value

    @property
    def quat(self):
        """Quaternion [qw, qx, qy, qz; body -> world]."""
        return self._state[6:10]

    @quat.setter
    def quat(self, value):
        self._state[6:10] = value

    @property
    def omega(self):
        """Angular velocity [rad/s; body frame]."""
        return self._state[10:13]

    @omega.setter
    def omega(self, value):
        self._state[10:13] = value

    def __repr__(self) -> str:
        return "State pos={}, vel={}, quat={}, omega={}".format(
            self.pos, self.vel, self.quat, self.omega
        )


class Quadrotor:
    """
    Basic rigid quadrotor model for crazyflie emulation
    """

    def __init__(self, state: State):
        self.mass = 0.034
        self.J = np.array([16.571710e-6, 16.655602e-6, 29.261652e-6])
        self.g = np.array([0, 0, -9.81])
        self.arm_length = 0.046
        self.prop_const = 0.006
        self.thrust2weight = 2.75
        self.HOVER_THRUST = self.mass * -self.g[2]
        self.max_thrust = self.HOVER_THRUST * self.thrust2weight
        self.max_torque = np.array(
            [
                self.max_thrust / 4 * self.arm_length,
                self.max_thrust / 4 * self.arm_length,
                2 * self.max_thrust / 4 * self.prop_const,
            ]
        )

        self.inv_J = 1 / self.J

        self.state: State = state
        self.step_counter = 0
        self._trajectory_window_length = 0  # initialize the trajectory window

        self._reference_trajectory = None

    @property
    def reference_trajectory(self):
        return self._reference_trajectory

    @reference_trajectory.setter
    def reference_trajectory(self, value):
        self._reference_trajectory = value

    @property
    def trajectory_window_length(self):
        return self._trajectory_window_length

    @trajectory_window_length.setter
    def trajectory_window_length(self, value):
        self._trajectory_window_length = value

    def get_trajectory_window(self):
        return self.reference_trajectory[ #
            self.step_counter : self.step_counter + self.trajectory_window_length
        ]

    def pre_process_action(self, action: np.ndarray) -> np.ndarray:

        force_torque_action = np.zeros(4)

        multipler = (self.thrust2weight - 1.0) if action[0] > 0.0 else 1.0
        normalized_thrust = action[0] * multipler + 1.0

        force_torque_action[0] = normalized_thrust * self.HOVER_THRUST
        force_torque_action[1:] = action[1:] * self.max_torque

        return force_torque_action

    def step(self, u, dt):
        u = self.pre_process_action(u)
        f_u = np.array([0, 0, u[0]])
        tau_u = -u[1:]

        omega_next = (
            self.state.omega
            + self.inv_J
            * (np.cross(self.J * self.state.omega, self.state.omega) + tau_u)
            * dt
        )
        self.state.omega = omega_next

        omega_global = rotate(self.state.quat, self.state.omega)
        q_next = q_integrate(self.state.quat, omega_global, dt)
        q_next = normalise_quat(q_next)
        self.state.quat = q_next

        x_current = np.concatenate([self.state.pos, self.state.vel])

        # implement the state update using RK4
        k1 = self.f_state(x_current, self.state.quat, u)
        k2 = self.f_state(x_current + k1 * dt / 2, self.state.quat, u)
        k3 = self.f_state(x_current + k2 * dt / 2, self.state.quat, u)
        k4 = self.f_state(x_current + k3 * dt, self.state.quat, u)

        x_next = x_current + (k1 + 2 * k2 + 2 * k3 + k4) * dt / 6

        self.state.pos = x_next[:3]
        self.state.vel = x_next[3:]
        self.step_counter += 1

    def f_state(self, x, quat, u):
        # returns the state derivative
        f_u = np.array([0, 0, u[0]])
        f_pos = x[3:]
        f_vel = self.g + rotate(quat, f_u / self.mass)
        return np.concatenate([f_pos, f_vel])
