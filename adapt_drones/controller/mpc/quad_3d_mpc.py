""" Implementation of the MPC for quadrotors."""

import numpy as np
from typing import Union

from adapt_drones.controller.mpc.quad_3d_optim import Quad3DOptimizer
from adapt_drones.controller.mpc.quad_3d import Quadrotor3D
from adapt_drones.utils.mpc_utils import simulate_plant


class Quad3DMPC:

    def __init__(
        self,
        quad: Quadrotor3D,
        t_horizon=1.0,
        n_nodes=5,
        q_cost=None,
        r_cost=None,
        optimization_dt=5e-2,
        simulation_dt=5e-4,
        model_name="my_quad",
        q_mask=None,
        solver_options=None,
        acados_path_postfix: Union[str, None] = None,
    ):
        self.quad: Quadrotor3D = quad
        self.simulation_dt = simulation_dt
        self.optim_dt = optimization_dt

        self.motor_u = np.array([0.0, 0.0, 0.0, 0.0])

        self.n_nodes = n_nodes
        self.t_horizon = t_horizon

        self.quad_opt = Quad3DOptimizer(
            quad=quad,
            t_horizon=t_horizon,
            nodes=n_nodes,
            q_cost=q_cost,
            r_cost=r_cost,
            q_mask=q_mask,
            solver_options=solver_options,
            model_name=model_name,
            acados_path_postfix=acados_path_postfix,
        )

    def clear(self):
        self.quad_opt.clear_acados_models()

    def get_state(self):
        """
        Returns the state of the drone, with the angle described as a wxyz quaternion
        :return: 13x1 array with the drone state: [p_xyz, a_wxyz, v_xyz, r_xyz]
        """
        x = np.expand_dims(self.quad.get_state(quaternion=True, stacked=True), 1)
        return x

    def set_reference(self, x_reference, u_reference=None):
        """
        Sets a target state for the MPC optimizer
        :param x_reference: list with 4 sub-components (position, angle quaternion, velocity, body rate). If these four
        are lists, then this means a single target point is used. If they are Nx3 and Nx4 (for quaternion) numpy arrays,
        then they are interpreted as a sequence of N tracking points.
        :param u_reference: Optional target for the optimized control inputs
        """

        if isinstance(x_reference[0], list):
            # Target state is just a point
            return self.quad_opt.set_reference_state(x_reference, u_reference)
        else:
            # Target state is a sequence
            return self.quad_opt.set_reference_trajectory(x_reference, u_reference)

    def optimize(self, use_model=0, return_x=False):
        """
        Runs MPC optimization to reach the pre-set target.
        :param use_model: Integer. Select which dynamics model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.

        :return: 4*m vector of optimized control inputs with the format: [u_1(0), u_2(0), u_3(0), u_4(0), u_1(1), ...,
        u_3(m-1), u_4(m-1)]. If return_x is True, will also return a vector of shape N+1 x 13 containing the optimized
        state prediction.
        """

        quad_current_state = self.quad.get_state(quaternion=True, stacked=True)

        # Remove rate state for simplified model NLP
        out_out = self.quad_opt.run_optimization(
            quad_current_state,
            use_model=use_model,
            return_x=return_x,
        )
        return out_out

    def simulate(self, ref_u):
        """
        Runs the simulation step for the dynamics model of the quadrotor 3D.

        :param ref_u: 4-length reference vector of control inputs
        """

        # Simulate step
        self.quad.update(ref_u, self.simulation_dt)

    # def simulate_plant(self, w_opt, t_horizon=None, dt_vec=None, progress_bar=False):
    #     """
    #     Given a sequence of n inputs, evaluates the simulated discrete-time plant model n steps into the future. The
    #     current drone state will not be changed by calling this method.
    #     :param w_opt: sequence of control n x m control inputs, where n is the number of steps and m is the
    #     dimensionality of a control input.
    #     :param t_horizon: time corresponding to the duration of the n control inputs. In the case that the w_opt comes
    #     from an MPC optimization, this parameter should be the MPC time horizon.
    #     :param dt_vec: a vector of timestamps, the same length as w_opt, corresponding to the total time each input is
    #     applied.
    #     :param progress_bar: boolean - whether to show a progress bar on the console or not.
    #     :return: the sequence of simulated quadrotor states.
    #     """

    #     if t_horizon is None and dt_vec is None:
    #         t_horizon = self.t_horizon
    #     print("IS THIS FUNCTRION BEING CALLED?")
    #     return simulate_plant(
    #         self.quad,
    #         w_opt,
    #         simulation_dt=self.simulation_dt,
    #         simulate_func=self.simulate,
    #         t_horizon=t_horizon,
    #         dt_vec=dt_vec,
    #         progress_bar=progress_bar,
    #     )

    @staticmethod
    def reshape_input_sequence(u_seq):
        """
        Reshapes the an output trajectory from the 1D format: [u_0(0), u_1(0), ..., u_0(n-1), u_1(n-1), ..., u_m-1(n-1)]
        to a 2D n x m array.
        :param u_seq: 1D input sequence
        :return: 2D input sequence, were n is the number of control inputs and m is the dimension of a single input.
        """

        k = np.arange(u_seq.shape[0] / 4, dtype=int)
        u_seq = np.atleast_2d(u_seq).T if len(u_seq.shape) == 1 else u_seq
        u_seq = np.concatenate(
            (u_seq[4 * k], u_seq[4 * k + 1], u_seq[4 * k + 2], u_seq[4 * k + 3]), 1
        )
        return u_seq

    def reset(self):
        return
