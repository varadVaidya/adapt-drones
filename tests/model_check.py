import numpy as np
import mujoco
import pkg_resources


def diff_model(model1, model2):
    skip_keys = ["names", "text_data", "paths"]
    struct_keys = ["stat", "opt", "vis"]
    for key in dir(model1):
        if (
            not key.startswith("__")
            and not callable(getattr(model1, key))
            and key not in skip_keys
            and key not in struct_keys
        ):
            try:
                if type(getattr(model1, key)) == np.ndarray:
                    check = np.all(getattr(model1, key) == getattr(model2, key))
                else:
                    check = getattr(model1, key) == getattr(model2, key)
                if not check:
                    print("diff", key)
                    print(getattr(model1, key))
                    print("\n ===================== \n")
                    print(getattr(model2, key))
                    print(
                        "\n =====================#################===================== \n"
                    )
            except Exception as e:
                print(getattr(model1, key))
                print("\n ===================== \n")
                print(getattr(model2, key))
    # for struct in struct_keys:
    #     for key in dir(getattr(model1, struct)):
    #         if not key.startswith("__") and not callable(
    #             getattr(getattr(model1, struct), key)
    #         ):
    #             if type(getattr(getattr(model1, struct), key)) == np.ndarray:
    #                 check = np.all(
    #                     getattr(getattr(model1, struct), key)
    #                     == getattr(getattr(model2, struct), key)
    #                 )
    #             else:
    #                 check = getattr(getattr(model1, struct), key) == getattr(
    #                     getattr(model2, struct), key
    #                 )
    #             if not check:
    #                 print("diff", struct, key)


if __name__ == "__main__":
    xml_file_1 = pkg_resources.resource_filename("adapt_drones", "assets/quad.xml")
    xml_file_2 = pkg_resources.resource_filename("adapt_drones", "assets/quad_rm.xml")

    model_1: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_file_1)
    data_1: mujoco.MjData = mujoco.MjData(model_1)
    drone_id = data_1.body("quad").id
    model_2: mujoco.MjModel = mujoco.MjModel.from_xml_path(xml_file_2)
    data_2: mujoco.MjData = mujoco.MjData(model_2)

    model_1.body_mass[drone_id] = 0.046
    model_1.body_inertia[drone_id] = np.array([4.28e-5, 4.28e-5, 8.36e-5])

    diff_model(model_1, model_2)

    print("###########################@@@@@@@@@@@@@@@@@@@@@@@@@##############")
    print("###########################@@@@@@@@@@@@@@@@@@@@@@@@@##############")
    print("###########################@@@@@@@@@@@@@@@@@@@@@@@@@##############")
    print("###########################@@@@@@@@@@@@@@@@@@@@@@@@@##############")
    print("###########################@@@@@@@@@@@@@@@@@@@@@@@@@##############")

    mujoco.mj_setConst(model_1, data_1)
    mujoco.mj_setConst(model_2, data_2)
    diff_model(model_1, model_2)
    # mujoco.mj_forward(model_1, data_1)
    # mujoco.mj_forward(model_1, data_1)
    # mujoco.mj_forward(model_1, data_1)
    # mujoco.mj_forward(model_1, data_1)

    # mujoco.mj_forward(model_2, data_2)
    # mujoco.mj_forward(model_2, data_2)
    # mujoco.mj_forward(model_2, data_2)
    # mujoco.mj_forward(model_2, data_2)
    # mujoco.mj_forward(model_2, data_2)

    # diff_model(model_1, model_2)
