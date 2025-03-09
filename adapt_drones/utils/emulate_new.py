import numpy as np

## quaternion utils for rotation. NOTE: Keeping all of the functions in the file different from the original code
## This tbh doesn't matter much, but i want to keep the emulation code independent of the original code


def unit_quat(q):
    """
    Normalizes a quaternion to be unit modulus.
    :param q: 4-dimensional numpy array or CasADi object
    :return:
    """
    if isinstance(q, np.ndarray):
        # if (q == np.zeros(4)).all():
        #     q = np.array([1, 0, 0, 0])
        q_norm = np.sqrt(np.sum(q**2))

        return q / q_norm


def q_mult(q1, q2):
    a = q1[0] * q2[0] - q1[1] * q2[1] - q1[2] * q2[2] - q1[3] * q2[3]
    b = q1[0] * q2[1] + q1[1] * q2[0] + q1[2] * q2[3] - q1[3] * q2[2]
    c = q1[0] * q2[2] - q1[1] * q2[3] + q1[2] * q2[0] + q1[3] * q2[1]
    d = q1[0] * q2[3] + q1[1] * q2[2] - q1[2] * q2[1] + q1[3] * q2[0]

    return np.array([a, b, c, d])


def quat_to_vel(quat, dt):

    axis = quat[1:]

    sin_a_2 = np.linalg.norm(axis)
    if sin_a_2 > 0:
        axis /= sin_a_2

    angle = 2 * np.arctan2(sin_a_2, quat[0])

    angle = angle if angle <= np.pi else angle - 2 * np.pi

    return axis * angle / dt


def conj_quat(q):
    return np.array([q[0], -q[1], -q[2], -q[3]])


def sub_quat(qa, qb):
    qb = conj_quat(qb)
    qdiff = q_mult(qb, qa)

    return quat_to_vel(qdiff, 1.0)


def v_dot_q(v, q):
    rot_mat = q_to_rot_mat(q)
    if isinstance(q, np.ndarray):
        return rot_mat.dot(v)


def q_to_rot_mat(q):
    qw, qx, qy, qz = q[0], q[1], q[2], q[3]

    if isinstance(q, np.ndarray):
        rot_mat = np.array(
            [
                [
                    1 - 2 * (qy**2 + qz**2),
                    2 * (qx * qy - qw * qz),
                    2 * (qx * qz + qw * qy),
                ],
                [
                    2 * (qx * qy + qw * qz),
                    1 - 2 * (qx**2 + qz**2),
                    2 * (qy * qz - qw * qx),
                ],
                [
                    2 * (qx * qz - qw * qy),
                    2 * (qy * qz + qw * qx),
                    1 - 2 * (qx**2 + qy**2),
                ],
            ]
        )
        return rot_mat


def skew_symmetric(v):
    """
    Computes the skew-symmetric matrix of a 3D vector (PAMPC version)

    :param v: 3D numpy vector or CasADi MX
    :return: the corresponding skew-symmetric matrix of v with the same data type as v
    """

    if isinstance(v, np.ndarray):
        return np.array(
            [
                [0, -v[0], -v[1], -v[2]],
                [v[0], 0, v[2], -v[1]],
                [v[1], -v[2], 0, v[0]],
                [v[2], v[1], -v[0], 0],
            ]
        )
    else:
        raise NotImplementedError(
            "Skew-symmetric matrix not implemented for data type {}".format(type(v))
        )


class State:
    """Class that stores the state of a UAV as used in the simulator interface."""

    def __init__(
        self,
        pos=np.zeros(3),
        vel=np.zeros(3),
        quat=np.array([1, 0, 0, 0]),
        omega=np.zeros(3),
    ):
        self.pos = pos
        self.vel = vel
        self.quat = quat
        self.omega = omega

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

        h = np.cos(np.pi / 4) * self.arm_length
        self.x_f = np.array([h, -h, -h, h])
        self.y_f = np.array([-h, -h, h, h])
        self.z_l_tau = np.array(
            [-self.prop_const, self.prop_const, -self.prop_const, self.prop_const]
        )

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
        self.trajectory_window_length = 0  # initialize the trajectory window

        self.reference_trajectory = None

    def get_trajectory_window(self):
        return self.reference_trajectory[  #
            self.step_counter : self.step_counter + self.trajectory_window_length
        ]

    def pre_process_action(self, action: np.ndarray) -> np.ndarray:

        force_torque_action = np.zeros(4)
        action = np.clip(action, -1.0, 1.0)
        multipler = (self.thrust2weight - 1.0) if action[0] > 0.0 else 1.0
        normalized_thrust = action[0] * multipler + 1.0

        force_torque_action[0] = normalized_thrust * self.HOVER_THRUST
        force_torque_action[1:] = action[1:] * self.max_torque

        return force_torque_action

    def set_state(self, *args):
        if len(args) != 0:
            assert len(args) == 13
            self.state.pos = args[0:3]
            self.state.quat = args[3:7]
            self.state.vel = args[7:10]
            self.state.omega = args[10:13]

    def get_state(self):
        return [self.state.pos, self.state.quat, self.state.vel, self.state.omega]

    def step(self, u, dt):
        u = self.pre_process_action(u)
        u[1:] = -u[1:]

        x = self.get_state()
        # RK4 integration
        k1 = [
            self.f_pos(x),
            self.f_att(x),
            self.f_vel(x, u),
            self.f_rate(x, u),
        ]
        x_aux = [x[i] + dt / 2 * k1[i] for i in range(4)]

        k2 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x, u),
            self.f_rate(x, u),
        ]
        x_aux = [x[i] + dt / 2 * k2[i] for i in range(4)]

        k3 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x, u),
            self.f_rate(x, u),
        ]
        x_aux = [x[i] + dt * k3[i] for i in range(4)]

        k4 = [
            self.f_pos(x_aux),
            self.f_att(x_aux),
            self.f_vel(x, u),
            self.f_rate(x, u),
        ]
        x = [
            x[i]
            + dt
            * (
                1.0 / 6.0 * k1[i]
                + 2.0 / 6.0 * k2[i]
                + 2.0 / 6.0 * k3[i]
                + 1.0 / 6.0 * k4[i]
            )
            for i in range(4)
        ]

        # Ensure unit quaternion
        x[1] = unit_quat(x[1])
        self.state.pos, self.state.quat, self.state.vel, self.state.omega = x
        self.step_counter += 1

    def f_pos(self, x):
        """
        Time-derivative of the position vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: position differential increment (vector): d[pos_x; pos_y]/dt
        """

        vel = x[2]
        return vel

    def f_att(self, x):
        """
        Time-derivative of the attitude in quaternion form
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :return: attitude differential increment (quaternion qw, qx, qy, qz): da/dt
        """

        rate = x[3]
        angle_quaternion = x[1]

        return 1 / 2 * skew_symmetric(rate).dot(angle_quaternion)

    def f_vel(self, x, u):
        """
        Time-derivative of the velocity vector
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [ Thrust, Torque_x, Torque_y, Torque_z ]
        :param f_d: disturbance force vector (3-dimensional)
        :return: 3D velocity differential increment (vector): d[vel_x; vel_y; vel_z]/dt
        """

        a_thrust = u[0] / self.mass
        a_thrust = np.array([0, 0, a_thrust])

        angle_quaternion = x[1]

        return self.g + v_dot_q(a_thrust, angle_quaternion)

    def f_rate(self, x, u):
        """
        Time-derivative of the angular rate
        :param x: 4-length array of input state with components: 3D pos, quaternion angle, 3D vel, 3D rate
        :param u: control input vector (4-dimensional): [ Thrust, Torque_x, Torque_y, Torque_z ]
        :return: angular rate differential increment (scalar): dr/dt
        """

        rate = x[3]
        return np.array(
            [
                1 / self.J[0] * (u[1] + (self.J[1] - self.J[2]) * rate[1] * rate[2]),
                1 / self.J[1] * (u[2] + (self.J[2] - self.J[0]) * rate[2] * rate[0]),
                1 / self.J[2] * (u[3] + (self.J[0] - self.J[1]) * rate[0] * rate[1]),
            ]
        )
