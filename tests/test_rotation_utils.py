import numpy as np
import mujoco
import random
import pkg_resources
import pyquaternion

from adapt_drones.utils.rotation import *


def test_mujoco_variables(seed=0):
    xml_path = pkg_resources.resource_filename("adapt_drones", "assets/quad.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)

    data = mujoco.MjData(model)

    rng = np.random.default_rng(seed=seed)
    data.ctrl = rng.random(model.nu)

    lvel = np.zeros(6)
    for _ in range(100):
        data.ctrl = rng.random(4)
        mujoco.mj_step(model, data)
        mujoco.mj_objectVelocity(
            model,
            data,
            mujoco.mjtObj.mjOBJ_BODY,
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "quad"),
            lvel,
            1,
        )
        calc_lvel = object_velocity(
            data.xipos[1], data.xmat[1].reshape(3, 3), data.cvel[1]
        )
        rot = np.zeros(9)
        assert np.allclose(lvel, calc_lvel)
        # mujoco.mju_quat2Mat(rot, data.xquat[1, 3:7])
        # print(data.qpos[3:7], data.xquat[1])
        # assert np.allclose(data.qpos[3:7], data.xquat[1], atol=1e-4, rtol=1e-4)
        # assert np.allclose(
        #     data.xmat[1].reshape(3, 3),
        #     rot.reshape(3, 3),
        # )

    print("DEBUG")


def test_spatial_velocity(seed=0):

    rng = np.random.default_rng(seed=seed)

    result_mujoco = np.zeros(6)
    vec = rng.random(6)
    mat = euler2mat(rng.random(3))
    flag_force = 0
    new_pos = rng.random(3)
    old_pos = new_pos

    mujoco.mju_transformSpatial(
        result_mujoco, vec, flag_force, new_pos, old_pos, mat.reshape(9)
    )

    result_python = transform_spatial(vec, flag_force, new_pos, old_pos, mat)

    print(result_mujoco, result_python)

    print("old vector: ", vec)
    print("Transformed vector: ", result_python)

    print(np.allclose(result_mujoco, result_python))


def test_quaternion(seed):
    rng = np.random.default_rng(seed=seed)

    rotation_matrix = euler2mat(rng.random(3))
    py_qaut = pyquaternion.Quaternion(matrix=rotation_matrix)
    mujoco_quat = np.zeros(4)
    mujoco.mju_mat2Quat(mujoco_quat, rotation_matrix.reshape(9))

    print(np.allclose(py_qaut.elements, mujoco_quat))


def fluid_force_model(mass, inertia, wind, pos, rot, vel, rho, beta):

    factor = 3 / (2 * mass)

    r_x = np.sqrt(factor * (inertia[1] + inertia[2] - inertia[0]))
    r_y = np.sqrt(factor * (inertia[2] + inertia[0] - inertia[1]))
    r_z = np.sqrt(factor * (inertia[0] + inertia[1] - inertia[2]))

    # check if the radii are real numbers
    assert np.all(np.isreal([r_x, r_y, r_z]))
    r = np.array([r_x, r_y, r_z])

    local_vel = object_velocity(pos, rot, vel)

    wind6 = np.zeros(6)
    wind6[3:] = wind
    local_wind = transform_spatial(wind6, 0, pos, pos, rot)
    # print(local_wind)

    local_vel[3:] -= local_wind[3:]

    wind_force = np.zeros(3)
    wind_torque = np.zeros(3)

    # viscous force

    r_eq = np.sum(r) / 3
    wind_force += -6 * beta * np.pi * r_eq * local_vel[3:]
    wind_torque += -8 * beta * np.pi * r_eq**3 * local_vel[:3]

    # drag force
    prod_r = np.prod(r)
    wind_force += -2 * rho * (prod_r / r) * np.abs(local_vel[3:]) * local_vel[3:]
    sum_r_4 = np.sum(r**4)
    wind_torque += (
        (-1 / 2) * rho * r * (sum_r_4 - r**4) * np.abs(local_vel[:3]) * local_vel[:3]
    )

    force_world = np.dot(rot, wind_force)
    torque_world = np.dot(rot, wind_torque)

    force_torque_world = np.concatenate([force_world, torque_world])

    return force_torque_world


def test_fluid_force_model(seed=0):
    rng = np.random.default_rng(seed=seed)
    xml_path = pkg_resources.resource_filename("adapt_drones", "assets/quad.xml")

    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.density = 0
    # model.opt.viscosity = 0
    data = mujoco.MjData(model)

    mass = model.body_mass[1]
    inertia = model.body_inertia[1]
    print(mass, inertia)
    model.opt.wind = np.array([-1.3, 0.87, -0.999])
    wind = model.opt.wind
    print(wind)

    rot = np.zeros(9)
    rotation_matrix = np.eye(3)
    for _ in range(700):
        data.ctrl = (
            np.array([(-2 * mass * model.opt.gravity[2]) + rng.random() + 10, 0, 0, 0])
            + rng.random(4) * 3
        )
        mujoco.mj_step(model, data)
        pos = data.xipos[1]
        vel = data.cvel[1]
        rotation_matrix = data.ximat[1].reshape(3, 3)
        # rotation_matrix = mujoco.mju_quat2Mat(rot, data.qpos[3:7])
        rho = model.opt.density
        # print(rho)
        beta = model.opt.viscosity
        # print(beta)

        print("---------------------------------------------------")
        forcetorque = fluid_force_model(
            mass, inertia, wind, pos, rotation_matrix, vel, rho, beta
        )

        mujoco_forceatorque = data.qfrc_passive

        # mujoco.mju_quat2Mat(rot, data.xquat[1])

        # print(vel)
        print(data.qpos[:3])
        print(data.qvel[:3])
        print(forcetorque)
        print(data.qfrc_fluid)
        print("---------------------------------------------------")


if __name__ == "__main__":
    seed = -1
    seed = random.randint(0, 2**32 - 1) if seed == -1 else seed
    # test_spatial_velocity(seed)

    # test_mujoco_variables(seed)
    # test_quaternion(seed)
    test_fluid_force_model(seed)
