import numpy as np

L_MIN, L_MAX = [0.046, 0.220]
M_MIN, M_MAX = [0.205, 1.841]
I_XX_MIN, I_XX_MAX = [1.23e-4, 1.75e-2]
I_YY_MIN, I_YY_MAX = [1.23e-4, 1.75e-2]
I_ZZ_MIN, I_ZZ_MAX = [2.10e-4, 3.40e-2]
tau_to_F_MIN, tau_to_F_MAX = [0.0051, 0.0170]
T_to_rpm2_MIN, T_to_rpm2_MAX = [3.88e-8, 8.40e-6]
BODY_DRAG_MIN, BODY_DRAG_MAX = [0, 0.74]
MOTOR_SPEED_MIN, MOTOR_SPEED_MAX = [800, 8044]


class Dynamics:

    def __init__(self, seed, c, do_random=True):
        self.c = c
        self.rng = seed
        self.do_random = do_random
        pass

    def random_output_dynamics(self, dynamics):
        return (
            self.rng.uniform(dynamics * 0.8, dynamics * 1.2)
            if self.do_random
            else dynamics
        )

    def length_scale(self):
        self.l = L_MIN + self.c * (L_MAX - L_MIN)
        # self.l = self.random_output_dynamics(L_MIN + self.c * (L_MAX - L_MIN))
        return self.l

    def mass_scale(self):
        cm = (self.l**3 - L_MIN**3) / (L_MAX**3 - L_MIN**3)
        return self.random_output_dynamics(M_MIN + cm * (M_MAX - M_MIN))

    def ixx_yy_scale(self):
        cJ = (self.l**5 - L_MIN**5) / (L_MAX**5 - L_MIN**5)
        return self.random_output_dynamics(I_XX_MIN + cJ * (I_XX_MAX - I_XX_MIN))

    def izz_scale(self):
        cJ = (self.l**5 - L_MIN**5) / (L_MAX**5 - L_MIN**5)
        return self.random_output_dynamics(I_ZZ_MIN + cJ * (I_ZZ_MAX - I_ZZ_MIN))

    def thrust_to_motor_speed(self):
        cf = T_to_rpm2_MIN * (T_to_rpm2_MAX / T_to_rpm2_MIN) ** self.c
        return self.random_output_dynamics(cf)

    def torque_to_thrust(self):
        # c_tau = (self.l - L_MIN) / (L_MAX - L_MIN)
        # return self.random_output_dynamics(
        #     tau_to_F_MIN + c_tau * (tau_to_F_MAX - tau_to_F_MIN)
        # )
        return self.random_output_dynamics(
            tau_to_F_MIN + self.c * (tau_to_F_MAX - tau_to_F_MIN)
        )

    def motor_speed_scale(self):
        # c_motor = (self.l - L_MIN) / (L_MAX - L_MIN)
        # return self.random_output_dynamics(
        #     MOTOR_SPEED_MIN + c_motor * (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN)
        # )
        # return self.random_output_dynamics(
        #     MOTOR_SPEED_MIN + self.c * (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN)
        # )

        # should be inverse scaling
        # c_motor = (self.l - L_MIN) / (L_MAX - L_MIN)
        # return self.random_output_dynamics(
        #     MOTOR_SPEED_MIN + (1 - c_motor) * (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN)
        # )
        return self.random_output_dynamics(
            MOTOR_SPEED_MIN + (1 - self.c) * (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN)
        )


class MinMaxDynamics:
    """
    A class to store the minimum and maximum values for each dynamics
    for the xadapt experiment.
    """

    def __init__(self, seed, c, do_random=True):
        self.c = c
        self.rng = seed
        self.do_random = do_random  # does nothing for now
        pass

    def output_min_max(self, dynamics):
        # return (
        #     self.rng.uniform(dynamics * 0.8, dynamics * 1.2)
        #     if self.do_random
        #     else dynamics
        # )

        return [dynamics * 0.8, dynamics * 1.2]

    def length_scale(self):
        l = L_MIN + self.c * (L_MAX - L_MIN)
        # l = self.output_min_max(L_MIN + self.c * (L_MAX - L_MIN))
        self.l_min, self.l_max = l, l
        return [self.l_min, self.l_max]

    def mass_scale(self):
        cm_min = (self.l_min**3 - L_MIN**3) / (L_MAX**3 - L_MIN**3)
        cm_max = (self.l_max**3 - L_MIN**3) / (L_MAX**3 - L_MIN**3)

        mass_min = self.output_min_max(M_MIN + cm_min * (M_MAX - M_MIN))[0]
        mass_max = self.output_min_max(M_MIN + cm_max * (M_MAX - M_MIN))[1]

        return [mass_min, mass_max]

    def ixx_yy_scale(self):

        cJ_min = (self.l_min**5 - L_MIN**5) / (L_MAX**5 - L_MIN**5)
        cJ_max = (self.l_max**5 - L_MIN**5) / (L_MAX**5 - L_MIN**5)

        ixx_min = self.output_min_max(I_XX_MIN + cJ_min * (I_XX_MAX - I_XX_MIN))[0]
        ixx_max = self.output_min_max(I_XX_MIN + cJ_max * (I_XX_MAX - I_XX_MIN))[1]

        return [ixx_min, ixx_max]
        # cJ = (self.l**5 - L_MIN**5) / (L_MAX**5 - L_MIN**5)
        # return self.random_output_dynamics(I_XX_MIN + cJ * (I_XX_MAX - I_XX_MIN))

    def izz_scale(self):
        cJ_min = (self.l_min**5 - L_MIN**5) / (L_MAX**5 - L_MIN**5)
        cJ_max = (self.l_max**5 - L_MIN**5) / (L_MAX**5 - L_MIN**5)

        izz_min = self.output_min_max(I_ZZ_MIN + cJ_min * (I_ZZ_MAX - I_ZZ_MIN))[0]
        izz_max = self.output_min_max(I_ZZ_MIN + cJ_max * (I_ZZ_MAX - I_ZZ_MIN))[1]

        return [izz_min, izz_max]
        # cJ = (self.l**5 - L_MIN**5) / (L_MAX**5 - L_MIN**5)
        # return self.random_output_dynamics(I_ZZ_MIN + cJ * (I_ZZ_MAX - I_ZZ_MIN))

    def thrust_to_motor_speed(self):
        cf = T_to_rpm2_MIN * (T_to_rpm2_MAX / T_to_rpm2_MIN) ** self.c
        return self.random_output_dynamics(cf)

    def torque_to_thrust(self):
        # c_tau = (self.l - L_MIN) / (L_MAX - L_MIN)
        # return self.random_output_dynamics(
        #     tau_to_F_MIN + c_tau * (tau_to_F_MAX - tau_to_F_MIN)
        # )

        return self.output_min_max(
            tau_to_F_MIN + self.c * (tau_to_F_MAX - tau_to_F_MIN)
        )

    def motor_speed_scale(self):
        # c_motor = (self.l - L_MIN) / (L_MAX - L_MIN)
        # return self.random_output_dynamics(
        #     MOTOR_SPEED_MIN + c_motor * (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN)
        # )
        # return self.random_output_dynamics(
        #     MOTOR_SPEED_MIN + self.c * (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN)
        # )

        # should be inverse scaling
        # c_motor = (self.l - L_MIN) / (L_MAX - L_MIN)
        # return self.random_output_dynamics(
        #     MOTOR_SPEED_MIN + (1 - c_motor) * (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN)
        # )
        return self.random_output_dynamics(
            MOTOR_SPEED_MIN + (1 - self.c) * (MOTOR_SPEED_MAX - MOTOR_SPEED_MIN)
        )
