""" Implementation of the nonlinear optimizer for MPC. """

import os
import sys
import shutil

from typing import List, Dict, Union
import casadi as cs
import numpy as np
from copy import copy
from acados_template import AcadosOcp, AcadosOcpSolver, AcadosModel

from adapt_drones.controller.mpc.quad_3d import Quadrotor3D
from adapt_drones.utils.mpc_utils import (
    skew_symmetric,
    v_dot_q,
    safe_mkdir_recursive,
    quaternion_inverse,
)
from adapt_drones.utils.mpc_utils import discretize_dynamics_and_cost


class Quad3DOptimizer:
    def __init__(
        self,
        quad: Quadrotor3D,
        t_horizon: int = 1,
        nodes: int = 20,
        q_cost: Union[np.ndarray, None] = None,
        r_cost: Union[np.ndarray, None] = None,
        q_mask: Union[np.ndarray, None] = None,
        model_name: str = "quad_3d",
        solver_options: Union[Dict, None] = None,
        acados_path_postfix: Union[str, None] = None,
    ):
        # Weighted squared error loss function q = (p_xyz, a_xyz, v_xyz, r_xyz), r = (u1, u2, u3, u4)
        if q_cost is None:
            q_cost = np.array(
                [10, 10, 10, 0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
            )
        if r_cost is None:
            r_cost = np.array([0.1, 0.1, 0.1, 0.1])

        self.T = t_horizon  # Time horizon
        self.N = nodes  # number of control nodes within horizon

        self.quad: Quadrotor3D = quad

        self.max_u = quad.max_input_value
        self.min_u = quad.min_input_value

        # Declare model variables
        self.p = cs.MX.sym("p", 3)  # position
        self.q = cs.MX.sym("a", 4)  # angle quaternion (wxyz)
        self.v = cs.MX.sym("v", 3)  # velocity
        self.r = cs.MX.sym("r", 3)  # angle rate

        # Full state vector (13-dimensional)
        self.x = cs.vertcat(self.p, self.q, self.v, self.r)
        self.state_dim = 13

        # Control input vector
        u1 = cs.MX.sym("u1")
        u2 = cs.MX.sym("u2")
        u3 = cs.MX.sym("u3")
        u4 = cs.MX.sym("u4")
        self.u = cs.vertcat(u1, u2, u3, u4)

        # Nominal model equations symbolic function

        self.quad_x_dot_nominal = self.quad_dynamics()
        # Linearized model dynamics symbolic function
        self.quad_xdot_jac = self.linearized_quad_dynamics()

        # Initialize objective function, 0 target state and integration equations
        self.L = None
        self.target = None

        # Build full model. Will have 13 variables.
        acados_model, dynamics_equations = self.acados_setup_model(
            self.quad_x_dot_nominal(x=self.x, u=self.u)["x_dot"], model_name
        )

        # Convert dynamics variables to functions of the state and input vectors
        self.quad_xdot = {}

        for dyn_model_idx in dynamics_equations.keys():
            dyn = dynamics_equations[dyn_model_idx]
            self.quad_xdot[dyn_model_idx] = cs.Function(
                "x_dot", [self.x, self.u], [dyn], ["x", "u"], ["x_dot"]
            )

        # ### Setup and compile Acados OCP solvers ### #
        self.acados_ocp_solver = {}

        # Add one more weight to the rotation (use quaternion norm weighting in acados)
        q_diagonal = np.concatenate(
            (q_cost[:3], np.mean(q_cost[3:6])[np.newaxis], q_cost[3:])
        )
        if q_mask is not None:
            q_mask = np.concatenate((q_mask[:3], np.zeros(1), q_mask[3:]))
            q_diagonal *= q_mask

        self.acados_models_dir = "acados_models"
        if acados_path_postfix is not None:
            self.acados_models_dir = self.acados_models_dir + "/" + acados_path_postfix
        safe_mkdir_recursive(os.path.join(os.getcwd(), self.acados_models_dir))

        for key, key_model in zip(acados_model.keys(), acados_model.values()):
            nx = key_model.x.size()[0]
            nu = key_model.u.size()[0]
            ny = nx + nu
            n_param = key_model.p.size()[0] if isinstance(key_model.p, cs.MX) else 0

            acados_source_path = os.environ["ACADOS_SOURCE_DIR"]

            # Create OCP object to formulate the optimization

            ocp = AcadosOcp()
            ocp.acados_include_path = acados_source_path + "/include"
            ocp.acados_lib_path = acados_source_path + "/lib"
            ocp.code_export_directory = (
                self.acados_models_dir + "/" + "c_generated_code"
            )
            ocp.model = key_model
            ocp.solver_options.N_horizon = self.N

            ocp.solver_options.tf = t_horizon

            # Initialaise partamteres
            ocp.dims.np = n_param
            ocp.parameter_values = np.zeros(n_param)

            ocp.cost.cost_type = "LINEAR_LS"
            ocp.cost.cost_type_e = "LINEAR_LS"

            ocp.cost.W = np.diag(np.concatenate((q_diagonal, r_cost)))
            ocp.cost.W_e = np.diag(q_diagonal)
            terminal_cost = (
                0
                if solver_options is None or not solver_options["terminal_cost"]
                else 1
            )
            ocp.cost.W_e *= terminal_cost

            ocp.cost.Vx = np.zeros((ny, nx))
            ocp.cost.Vx[:nx, :nx] = np.eye(nx)
            ocp.cost.Vu = np.zeros((ny, nu))
            ocp.cost.Vu[-4:, -4:] = np.eye(nu)

            ocp.cost.Vx_e = np.eye(nx)

            # Initial reference trajectory (will be overwritten)
            x_ref = np.zeros(nx)
            ocp.cost.yref = np.concatenate((x_ref, np.array([0.0, 0.0, 0.0, 0.0])))
            ocp.cost.yref_e = x_ref

            # Initial state (will be overwritten)
            ocp.constraints.x0 = x_ref

            # Set constraints
            ocp.constraints.lbu = np.array([self.min_u] * 4)
            ocp.constraints.ubu = np.array([self.max_u] * 4)
            ocp.constraints.idxbu = np.array([0, 1, 2, 3])

            # Solver options
            ocp.solver_options.qp_solver = "FULL_CONDENSING_HPIPM"
            ocp.solver_options.hessian_approx = "GAUSS_NEWTON"
            ocp.solver_options.integrator_type = "ERK"
            ocp.solver_options.print_level = 0
            ocp.solver_options.nlp_solver_type = (
                "SQP_RTI" if solver_options is None else solver_options["solver_type"]
            )

            # Compile acados OCP solver if necessary
            json_file = os.path.join(
                self.acados_models_dir, key_model.name + "_acados_ocp.json"
            )

            self.json_file = json_file
            self.acados_ocp_solver[key] = AcadosOcpSolver(
                ocp, json_file=json_file, verbose=False
            )

    def quad_dynamics(self):
        """
        Symbolic dynamics of the 3D quadrotor model. The state consists on: [p_xyz, a_wxyz, v_xyz, r_xyz]^T, where p
        stands for position, a for angle (in quaternion form), v for velocity and r for body rate. The input of the
        system is: [u_1, u_2, u_3, u_4], i.e. the activation of the four thrusters.


        :return: CasADi function that computes the analytical differential state dynamics of the quadrotor model.
        Inputs: 'x' state of quadrotor (6x1) and 'u' control input (2x1). Output: differential state vector 'x_dot'
        (6x1)
        """

        x_dot = cs.vertcat(
            self.p_dynamics(),
            self.q_dynamics(),
            self.v_dynamics(),
            self.w_dynamics(),
        )
        return cs.Function(
            "x_dot", [self.x[:13], self.u], [x_dot], ["x", "u"], ["x_dot"]
        )

    def linearized_quad_dynamics(self):
        """
        Jacobian J matrix of the linearized dynamics specified in the function quad_dynamics. J[i, j] corresponds to
        the partial derivative of f_i(x) wrt x(j).

        :return: a CasADi symbolic function that calculates the 13 x 13 Jacobian matrix of the linearized simplified
        quadrotor dynamics
        """

        jac = cs.MX(self.state_dim, self.state_dim)

        # Position derivatives
        jac[0:3, 7:10] = cs.diag(cs.MX.ones(3))

        # Angle derivatives
        jac[3:7, 3:7] = skew_symmetric(self.r) / 2
        jac[3, 10:] = 1 / 2 * cs.horzcat(-self.q[1], -self.q[2], -self.q[3])
        jac[4, 10:] = 1 / 2 * cs.horzcat(self.q[0], -self.q[3], self.q[2])
        jac[5, 10:] = 1 / 2 * cs.horzcat(self.q[3], self.q[0], -self.q[1])
        jac[6, 10:] = 1 / 2 * cs.horzcat(-self.q[2], self.q[1], self.q[0])

        # Velocity derivatives
        a_u = (
            (self.u[0] + self.u[1] + self.u[2] + self.u[3])
            * self.quad.max_thrust
            / self.quad.mass
        )
        jac[7, 3:7] = 2 * cs.horzcat(
            a_u * self.q[2], a_u * self.q[3], a_u * self.q[0], a_u * self.q[1]
        )
        jac[8, 3:7] = 2 * cs.horzcat(
            -a_u * self.q[1], -a_u * self.q[0], a_u * self.q[3], a_u * self.q[2]
        )
        jac[9, 3:7] = 2 * cs.horzcat(0, -2 * a_u * self.q[1], -2 * a_u * self.q[1], 0)

        # Rate derivatives
        jac[10, 10:] = (
            (self.quad.J[1] - self.quad.J[2])
            / self.quad.J[0]
            * cs.horzcat(0, self.r[2], self.r[1])
        )
        jac[11, 10:] = (
            (self.quad.J[2] - self.quad.J[0])
            / self.quad.J[1]
            * cs.horzcat(self.r[2], 0, self.r[0])
        )
        jac[12, 10:] = (
            (self.quad.J[0] - self.quad.J[1])
            / self.quad.J[2]
            * cs.horzcat(self.r[1], self.r[0], 0)
        )

        return cs.Function("J", [self.x, self.u], [jac])

    def acados_setup_model(self, nominal, model_name):
        """
        Builds an Acados symbolic models using CasADi expressions.
        :param model_name: name for the acados model. Must be different from previously used names or there may be
        problems loading the right model.
        :param nominal: CasADi symbolic nominal model of the quadrotor: f(self.x, self.u) = x_dot, dimensions 13x1.
        :return: Returns a total of three outputs, where m is the number of GP's in the GP ensemble, or 1 if no GP:
            - A dictionary of m AcadosModel of the GP-augmented quadrotor
            - A dictionary of m CasADi symbolic nominal dynamics equations with GP mean value augmentations (if with GP)
        :rtype: dict, dict, cs.MX
        """

        def fill_in_acados_model(x, u, p, dynamics, name):

            x_dot = cs.MX.sym("x_dot", dynamics.shape)
            f_impl = x_dot - dynamics

            # Dynamics model
            model = AcadosModel()
            model.f_expl_expr = dynamics
            model.f_impl_expr = f_impl
            model.x = x
            model.xdot = x_dot
            model.u = u
            model.p = p
            model.name = name

            return model

        acados_models = {}
        dynamics_equations = {}

        dynamics_equations[0] = nominal
        x_ = self.x
        dynamics_ = nominal

        acados_models[0] = fill_in_acados_model(
            x=x_, u=self.u, p=[], dynamics=dynamics_, name=model_name
        )

        return acados_models, dynamics_equations

    def clear_acados_models(self):
        """
        Removes previous stored acados models to avoid conflicts with new models.
        """
        json_file = self.json_file

        if os.path.exists(json_file):
            os.remove(json_file)
        c_code_dir = self.acados_models_dir + "/" + "c_generated_code"

        if os.path.exists(c_code_dir):
            shutil.rmtree(c_code_dir)

    def set_reference_state(self, x_target=None, u_target=None):
        """
        Sets the target state and pre-computes the integration dynamics with cost equations
        :param x_target: 13-dimensional target state (p_xyz, a_wxyz, v_xyz, r_xyz)
        :param u_target: 4-dimensional target control input vector (u_1, u_2, u_3, u_4)
        """

        if x_target is None:
            x_target = [[0, 0, 0], [1, 0, 0, 0], [0, 0, 0], [0, 0, 0]]
        if u_target is None:
            u_target = [0, 0, 0, 0]
        # Set new target state
        self.target = copy(x_target)

        ref = np.concatenate([x_target[i] for i in range(4)])
        #  Transform velocity to body frame
        v_b = v_dot_q(ref[7:10], quaternion_inverse(ref[3:7]))
        ref = np.concatenate((ref[:7], v_b, ref[10:]))

        idx = 0
        ref = np.concatenate((ref, u_target))

        for j in range(self.N):
            self.acados_ocp_solver[idx].set(j, "yref", ref)
            self.acados_ocp_solver[idx].set(self.N, "yref", ref[:-4])

        return idx

    def set_reference_trajectory(self, x_target, u_target):
        """
        Sets the reference trajectory and pre-computes the cost equations for each point in the reference sequence.
        :param x_target: Nx13-dimensional reference trajectory (p_xyz, angle_wxyz, v_xyz, rate_xyz). It is passed in the
        form of a 4-length list, where the first element is a Nx3 numpy array referring to the position targets, the
        second is a Nx4 array referring to the quaternion, two more Nx3 arrays for the velocity and body rate targets.
        :param u_target: Nx4-dimensional target control input vector (u1, u2, u3, u4)
        """

        if u_target is not None:
            assert (
                x_target[0].shape[0] == (u_target.shape[0] + 1)
                or x_target[0].shape[0] == u_target.shape[0]
            )

        # If not enough states in target sequence, append last state until required length is met
        while x_target[0].shape[0] < self.N + 1:
            x_target = [
                np.concatenate((x, np.expand_dims(x[-1, :], 0)), 0) for x in x_target
            ]
            if u_target is not None:
                u_target = np.concatenate(
                    (u_target, np.expand_dims(u_target[-1, :], 0)), 0
                )

        stacked_x_target = np.concatenate([x for x in x_target], 1)

        #  Transform velocity to body frame
        x_mean = stacked_x_target[int(self.N / 2)]
        v_b = v_dot_q(x_mean[7:10], quaternion_inverse(x_mean[3:7]))
        x_target_mean = np.concatenate((x_mean[:7], v_b, x_mean[10:]))

        idx = 0

        self.target = copy(x_target)

        for j in range(self.N):
            ref = stacked_x_target[j, :]
            ref = np.concatenate((ref, u_target[j, :]))
            self.acados_ocp_solver[idx].set(j, "yref", ref)
        # the last MPC node has only a state reference but no input reference
        self.acados_ocp_solver[idx].set(self.N, "yref", stacked_x_target[self.N, :])

        return idx

    def discretize_f_and_q(self, t_horizon, n, m=1, i=0, use_model=0):
        """
        Discretize the model dynamics and the pre-computed cost function if available.
        :param t_horizon: time horizon in seconds
        :param n: number of control steps until time horizon
        :param m: number of integration steps per control step
        :param i: Only used for trajectory tracking. Index of cost function to use.
        :param use_gp: Whether to use the dynamics with the GP correction or not.
        :param use_model: integer, select which model to use from the available options.
        :return: the symbolic, discretized dynamics. The inputs of the symbolic function are x0 (the initial state) and
        p, the control input vector. The outputs are xf (the updated state) and qf. qf is the corresponding cost
        function of the integration, which is calculated from the pre-computed discrete-time model dynamics (self.L)
        """

        dynamics = self.quad_xdot_nominal

        # Call with self.x_with_gp even if use_gp=False #TODO: see what this means
        return discretize_dynamics_and_cost(
            t_horizon, n, m, self.x, self.u, dynamics, self.L, i
        )

    def run_optimization(self, initial_state=None, use_model=0, return_x=False):
        """
        Optimizes a trajectory to reach the pre-set target state, starting from the input initial state, that minimizes
        the quadratic cost function and respects the constraints of the system

        :param initial_state: 13-element list of the initial state. If None, 0 state will be used
        :param use_model: integer, select which model to use from the available options.
        :param return_x: bool, whether to also return the optimized sequence of states alongside with the controls.
        :return: optimized control input sequence (flattened)
        """

        if initial_state is None:
            initial_state = [0, 0, 0] + [1, 0, 0, 0] + [0, 0, 0] + [0, 0, 0]

        # Set initial state.
        x_init = initial_state
        x_init = np.stack(x_init)

        # Set initial condition, equality constraint
        self.acados_ocp_solver[use_model].set(0, "lbx", x_init)
        self.acados_ocp_solver[use_model].set(0, "ubx", x_init)

        # Solve OCP
        self.acados_ocp_solver[use_model].solve()

        # Get u
        w_opt_acados = np.ndarray((self.N, 4))
        x_opt_acados = np.ndarray((self.N + 1, len(x_init)))
        x_opt_acados[0, :] = self.acados_ocp_solver[use_model].get(0, "x")
        for i in range(self.N):
            w_opt_acados[i, :] = self.acados_ocp_solver[use_model].get(i, "u")
            x_opt_acados[i + 1, :] = self.acados_ocp_solver[use_model].get(i + 1, "x")

        w_opt_acados = np.reshape(w_opt_acados, (-1))
        return w_opt_acados if not return_x else (w_opt_acados, x_opt_acados)

    def p_dynamics(self):
        return self.v

    def q_dynamics(self):
        return 1 / 2 * cs.mtimes(skew_symmetric(self.r), self.q)

    def v_dynamics(self):

        f_thrust = self.u * self.quad.max_thrust
        g = cs.vertcat(0.0, 0.0, 9.81)
        a_thrust = (
            cs.vertcat(0.0, 0.0, f_thrust[0] + f_thrust[1] + f_thrust[2] + f_thrust[3])
            / self.quad.mass
        )

        v_dynamics = v_dot_q(a_thrust, self.q) - g

        return v_dynamics

    def w_dynamics(self):
        f_thrust = self.u * self.quad.max_thrust

        y_f = cs.MX(self.quad.y_f)
        x_f = cs.MX(self.quad.x_f)
        c_f = cs.MX(self.quad.z_l_tau)
        return cs.vertcat(
            (
                cs.mtimes(f_thrust.T, y_f)
                + (self.quad.J[1] - self.quad.J[2]) * self.r[1] * self.r[2]
            )
            / self.quad.J[0],
            (
                -cs.mtimes(f_thrust.T, x_f)
                + (self.quad.J[2] - self.quad.J[0]) * self.r[2] * self.r[0]
            )
            / self.quad.J[1],
            (
                cs.mtimes(f_thrust.T, c_f)
                + (self.quad.J[0] - self.quad.J[1]) * self.r[0] * self.r[1]
            )
            / self.quad.J[2],
        )
