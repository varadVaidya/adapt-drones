from numba import jit
import numpy as np

_FLOAT_EPS = np.finfo(np.float64).eps
_EPS4 = _FLOAT_EPS * 4.0


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


@jit(nopython=True)
def mat2euler(mat):
    """Convert Rotation Matrix to Euler Angles.  See rotation.py for notes"""
    mat = np.asarray(mat, dtype=np.float64)
    # assert mat.shape[-2:] == (3, 3), "Invalid shape matrix {}".format(mat)

    cy = np.sqrt(mat[..., 2, 2] * mat[..., 2, 2] + mat[..., 1, 2] * mat[..., 1, 2])
    condition = cy > _EPS4
    euler = np.empty(mat.shape[:-1], dtype=np.float64)
    euler[..., 2] = np.where(
        condition,
        -np.arctan2(mat[..., 0, 1], mat[..., 0, 0]),
        -np.arctan2(-mat[..., 1, 0], mat[..., 1, 1]),
    )
    euler[..., 1] = np.where(
        condition, -np.arctan2(-mat[..., 0, 2], cy), -np.arctan2(-mat[..., 0, 2], cy)
    )
    euler[..., 0] = np.where(
        condition, -np.arctan2(mat[..., 1, 2], mat[..., 2, 2]), 0.0
    )
    return euler


@jit(nopython=True)
def euler2mat(euler):
    """Convert Euler Angles to Rotation Matrix.  See rotation.py for notes"""
    euler = np.asarray(euler, dtype=np.float64)
    # assert euler.shape[-1] == 3, "Invalid shaped euler {}".format(euler)

    ai, aj, ak = -euler[..., 2], -euler[..., 1], -euler[..., 0]
    si, sj, sk = np.sin(ai), np.sin(aj), np.sin(ak)
    ci, cj, ck = np.cos(ai), np.cos(aj), np.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk

    mat = np.empty(euler.shape[:-1] + (3, 3), dtype=np.float64)
    mat[..., 2, 2] = cj * ck
    mat[..., 2, 1] = sj * sc - cs
    mat[..., 2, 0] = sj * cc + ss
    mat[..., 1, 2] = cj * sk
    mat[..., 1, 1] = sj * ss + cc
    mat[..., 1, 0] = sj * cs - sc
    mat[..., 0, 2] = -sj
    mat[..., 0, 1] = cj * si
    mat[..., 0, 0] = cj * ci
    return mat


###=========== ROTATION UTILS THAT ARE DIRECT TRANSLATION OF MUJOCO CODE TO AID IN THE MPC IMPLEMENTATION ===========###


# equivalent to mju_transformSpatial
def transform_spatial(vec, flg_force, newpos, oldpos, rotnew2old=None):
    # Create output and intermediate arrays
    res = np.zeros(6)
    cros = np.zeros(3)
    dif = np.zeros(3)
    tran = np.copy(vec)

    # Compute difference between newpos and oldpos
    dif = newpos - oldpos

    # Apply translation
    if flg_force:
        # Cross product of dif and vec[3:6]
        cros = np.cross(dif, vec[3:6])
        tran[0:3] = vec[0:3] - cros
    else:
        # Cross product of dif and vec[0:3]
        cros = np.cross(dif, vec[0:3])
        tran[3:6] = vec[3:6] - cros

    # Apply rotation if provided
    if rotnew2old is not None:
        # Apply rotation to angular part (first 3 elements) and linear part (next 3 elements)
        res[0:3] = np.dot(rotnew2old.T, tran[0:3])
        res[3:6] = np.dot(rotnew2old.T, tran[3:6])
    else:
        # If no rotation is provided, just copy the translated vector
        res = np.copy(tran)

    return res


# equivalent to mj_object velocity only for the types and condition we care about
def object_velocity(pos, rot, vel):

    return transform_spatial(vel, 0, pos, pos, rot)
