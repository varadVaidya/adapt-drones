import numpy as np
import mujoco
import mujoco.renderer
from scipy.spatial.transform import Rotation


def add_visual_capsule(
    scene: mujoco.MjvScene, point1, point2, radius, rgba: np.ndarray
):

    if scene.ngeom >= scene.maxgeom:
        return
    scene.ngeom += 1
    mujoco.mjv_initGeom(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        rgba.astype(np.float32),
    )
    mujoco.mjv_makeConnector(
        scene.geoms[scene.ngeom - 1],
        mujoco.mjtGeom.mjGEOM_CAPSULE,
        radius,
        point1[0],
        point1[1],
        point1[2],
        point2[0],
        point2[1],
        point2[2],
    )


def modify_scene(scene: mujoco.MjvScene, position_trace, ref_position_trace=None):
    """Draw the position traces on the string

    Args:
        scene (mujoco.MjvScene):
        position_trace (deque): deque of positions
        ref_position_trace (deque, optional): deque of position. Defaults to None.
    """

    if len(position_trace) > 1:
        for i in range(len(position_trace) - 1):
            rgba = np.array([0.2, 1.0, 0.2, 0.75])
            radius = 0.0025
            point1 = position_trace[i]
            point2 = position_trace[i + 1]
            add_visual_capsule(scene, point1, point2, radius, rgba)

    if ref_position_trace is not None:
        if len(ref_position_trace) > 1:
            for i in range(len(ref_position_trace) - 1):
                rgba = np.array([1.0, 0.2, 0.2, 0.75])
                radius = 0.0025
                point1 = ref_position_trace[i]
                point2 = ref_position_trace[i + 1]
                add_visual_capsule(scene, point1, point2, radius, rgba)


# See https://github.com/microsoft/pylance-release/issues/3277
def cross2(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.cross(a, b)


# render vector is based on : https://github.com/iit-DLSLab/gym-quadruped/blob/2596a4571e3ba14acc73fe900b0927efd11c85d6/gym_quadruped/utils/mujoco/visual.py#L15-L16
# No LICENSE found in the original source as of 2024-10-04
def render_vector(
    scene: mujoco.MjvScene,
    vector: np.ndarray,
    pos: np.ndarray,
    scale: float,
    color: np.ndarray = np.array([1, 0, 0, 1]),
    geom_id: int = -1,
) -> int:
    """
    Function to render a vector in the Mujoco viewer.

    Args:
        viewer (Handle): The Mujoco viewer.
        vector (np.ndarray): The vector to render.
        pos (np.ndarray): The position of the base of vector.
        scale (float): The scale of the vector.
        color (np.ndarray): The color of the vector.
        geom_id (int, optional): The id of the geometry. Defaults to -1.
    Returns:
        int: The id of the geometry.
    """
    if geom_id < 0:
        # Instantiate a new geometry
        geom = mujoco.MjvGeom()
        geom.type = mujoco.mjtGeom.mjGEOM_ARROW
        scene.ngeom += 1
        geom_id = scene.ngeom - 1

    geom = scene.geoms[geom_id]

    # Define the a rotation matrix with the Z axis aligned with the vector direction
    vec_z = vector.squeeze() / np.linalg.norm(vector + 1e-5)
    # Define any orthogonal to z vector as the X axis using the Gram-Schmidt process
    rand_vec = np.random.rand(3)
    vec_x = rand_vec - (np.dot(rand_vec, vec_z) * vec_z)
    vec_x = vec_x / np.linalg.norm(vec_x)
    # Define the Y axis as the cross product of X and Z
    vec_y = cross2(vec_z, vec_x)

    ori_mat = Rotation.from_matrix(np.array([vec_x, vec_y, vec_z]).T).as_matrix()
    mujoco.mjv_initGeom(
        geom,
        type=mujoco.mjtGeom.mjGEOM_ARROW,
        size=np.asarray([0.01, 0.01, scale]),
        pos=pos,
        mat=ori_mat.flatten(),
        rgba=color,
    )
    geom.category = mujoco.mjtCatBit.mjCAT_DECOR
    geom.segid = -1
    geom.objid = -1

    return geom_id
