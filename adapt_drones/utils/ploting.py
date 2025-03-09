from dataclasses import dataclass, asdict
import numpy as np
from numpy import ndarray
import matplotlib.pyplot as plt
from matplotlib.offsetbox import AnchoredText


@dataclass
class TextonPlot:
    seed: str
    mass: str
    inertia: str
    wind: str
    com: str
    prop_const: str
    arm_length: str
    thrust2weight: str
    mean_error: str
    rms_error: str


def data_plot(
    t: ndarray,
    position: ndarray,
    goal_position: ndarray,
    velocity: ndarray,
    goal_velocity: ndarray,
    quaternion: ndarray,
    angular_velocity: ndarray,
    action: ndarray,
    plot_text: TextonPlot,
    save_prefix: str,
    save_path: str,
):
    assert len(t) == len(position), "Length of t and position are not same"
    pos_error = goal_position - position
    mean_error = np.mean(np.linalg.norm(pos_error, axis=1))
    rms_error = np.sqrt(np.mean(np.linalg.norm(pos_error, axis=1) ** 2))
    print("Mean Error:", mean_error)
    print("RMS Error:", rms_error)
    plot_text.mean_error = "Mean Error: " + str(mean_error)
    plot_text.rms_error = "RMS Error: " + str(rms_error)

    vel_error = goal_velocity - velocity

    plot_mosaic = [
        ["pos", "pos", "x-y_plot", "x-y_plot"],
        ["vel", "vel", "x-y_plot", "x-y_plot"],
        ["quat", "quat", "ang_vel", "ang_vel"],
        ["thrust", "thrust", "torque", "torque"],
    ]
    fig, axs = plt.subplot_mosaic(
        plot_mosaic,
        figsize=(30, 30),
        constrained_layout=True,
        per_subplot_kw={("x-y_plot",): {"projection": "3d"}},
    ) # type: ignore

    axs["pos"].plot(t, position[:, 0], "r", label="x")
    axs["pos"].plot(t, position[:, 1], "g", label="y")
    axs["pos"].plot(t, position[:, 2], "b", label="z")

    axs["pos"].plot(t, goal_position[:, 0], "r--", alpha=0.75, label="x_ref")
    axs["pos"].plot(t, goal_position[:, 1], "g--", alpha=0.75, label="y_ref")
    axs["pos"].plot(t, goal_position[:, 2], "b--", alpha=0.75, label="z_ref")
    axs["pos"].grid()
    axs["pos"].set_title("Position")
    axs["pos"].legend(ncol=2)

    axs["vel"].plot(t, velocity[:, 0], "r", label="vx")
    axs["vel"].plot(t, velocity[:, 1], "g", label="vy")
    axs["vel"].plot(t, velocity[:, 2], "b", label="vz")

    axs["vel"].plot(t, goal_velocity[:, 0], "r--", alpha=0.75, label="vx_ref")
    axs["vel"].plot(t, goal_velocity[:, 1], "g--", alpha=0.75, label="vy_ref")
    axs["vel"].plot(t, goal_velocity[:, 2], "b--", alpha=0.75, label="vz_ref")
    axs["vel"].grid()
    axs["vel"].set_title("Velocity")
    axs["vel"].legend(ncol=2)
    
    axs["quat"].plot(t, quaternion[:, 0], "r", label="w")
    axs["quat"].plot(t, quaternion[:, 1], "g", label="x")
    axs["quat"].plot(t, quaternion[:, 2], "b", label="y")
    axs["quat"].plot(t, quaternion[:, 3], "y", label="z")
    axs["quat"].grid()
    axs["quat"].set_title("Quaternion")
    axs["quat"].legend(ncol=2)
    
    axs["ang_vel"].plot(t, angular_velocity[:, 0], "r", label="wx")
    axs["ang_vel"].plot(t, angular_velocity[:, 1], "g", label="wy")
    axs["ang_vel"].plot(t, angular_velocity[:, 2], "b", label="wz")
    axs["ang_vel"].grid()
    axs["ang_vel"].set_title("Angular Velocity")
    axs["ang_vel"].legend(ncol=1)
    
    axs["thrust"].plot(t, action[:, 0], "r", label="thrust")
    axs["thrust"].grid()
    axs["thrust"].set_title("Thrust")
    axs["thrust"].legend(ncol=1)
    
    axs["torque"].plot(t, action[:, 1], "r", label="tx")
    axs["torque"].plot(t, action[:, 2], "g", label="ty")
    axs["torque"].plot(t, action[:, 3], "b", label="tz")
    axs["torque"].grid()
    axs["torque"].set_title("Torque")
    axs["torque"].legend(ncol=1)
    

    axs["x-y_plot"].plot(position[:, 0], position[:, 1], position[:, 2], "k")
    axs["x-y_plot"].plot(
        goal_position[:, 0], goal_position[:, 1], goal_position[:, 2], "y--"
    )
    axs["x-y_plot"].set_xlabel("x(m)")
    axs["x-y_plot"].set_ylabel("y(m)")
    axs["x-y_plot"].set_zlabel("z(m)")

    anchr_text = "\n".join("{}".format(v) for k, v in asdict(plot_text).items())
    anchr_text = AnchoredText(anchr_text, loc=2)
    axs["x-y_plot"].add_artist(anchr_text)

    plt.savefig(save_path + save_prefix + ".png")
