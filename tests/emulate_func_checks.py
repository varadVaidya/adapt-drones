import random
import numpy as np
import mujoco

from adapt_drones.utils.emulate import *


if __name__ == "__main__":
    seed = -1
    seed = seed if seed != -1 else random.randint(0, 2**32 - 1)

    rng = np.random.default_rng(seed)

    results = []

    for i in range(1_000_000):

        q_a = normalise_quat(rng.uniform(-1, 1, (4,)))
        q_b = normalise_quat(rng.uniform(-1, 1, (4,)))

        delta_ori = np.zeros(3)
        mujoco.mju_subQuat(delta_ori, q_a, q_b)

        calc_delta_ori = sub_quat(q_a, q_b)

        # print(calc_delta_ori, delta_ori)
        results.append(np.allclose(calc_delta_ori, delta_ori))

    results = np.array(results)
    print(f"Any false: {np.any(~results)}")
    print(f"Number of false: {np.sum(~results)}")
