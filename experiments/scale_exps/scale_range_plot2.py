import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-deep")
plt.rcParams["text.usetex"] = True
# latex settings for matplotlib
from dataclasses import dataclass
import tyro
from adapt_drones.cfgs.config import *
import matplotlib.gridspec as gridspec


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "laced-fire-32"
    seed: int = 20240915
    agent: str = "RMA_DATT"
    scale: bool = True


args = tyro.cli(Args)

cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    eval=True,
    run_name=args.run_name,
    agent=args.agent,
    scale=args.scale,
)


## --------------- TRAIN RANGE PLOT ----------------- ##
_sc_lengths = cfg.environment.scale_lengths
print(_sc_lengths)
arm_lengths = np.linspace(_sc_lengths[0], _sc_lengths[1], 100)

mass = np.polyval(cfg.scale.avg_mass_fit, arm_lengths)
mass_std = np.polyval(cfg.scale.std_mass_fit, arm_lengths)
mass_range = np.array([mass - mass_std, mass + mass_std])
mass_range[mass_range < 0] = 0

ixx = np.polyval(cfg.scale.avg_ixx_fit, arm_lengths)
ixx_std = np.polyval(cfg.scale.std_ixx_fit, arm_lengths)
ixx_range = np.array([ixx - ixx_std, ixx + ixx_std])
ixx_range[ixx_range < 0] = 0

iyy = np.polyval(cfg.scale.avg_iyy_fit, arm_lengths)
iyy_std = np.polyval(cfg.scale.std_iyy_fit, arm_lengths)
iyy_range = np.array([iyy - iyy_std, iyy + iyy_std])
iyy_range[iyy_range < 0] = 0

izz = np.polyval(cfg.scale.avg_izz_fit, arm_lengths)
izz_std = np.polyval(cfg.scale.std_izz_fit, arm_lengths)
izz_range = np.array([izz - izz_std, izz + izz_std])
izz_range[izz_range < 0] = 0

km_kf = np.polyval(cfg.scale.avg_km_kf_fit, arm_lengths)
km_kf_std = np.polyval(cfg.scale.std_km_kf_fit, arm_lengths)
km_kf_range = np.array([km_kf - km_kf_std, km_kf + km_kf_std])
km_kf_range[km_kf_range < 0] = 0


fig, axs = plt.subplots(
    5,
    1,
    figsize=(3, 6),
    sharex=True,
    height_ratios=[2, 2, 2, 2, 2],
    layout="compressed",
)
ax1, ax2, ax3, ax4, ax5 = axs


# fig = plt.figure(figsize=(10, 8))
# gs = fig.add_gridspec(2, 6)
# # gs = gridspec.GridSpec(2, 6)
# gs.update(hspace=0.1875, wspace=1, left=0.1, right=0.9, top=0.9, bottom=0.1)

# ax1 = plt.subplot(gs[0, :2])
# ax2 = plt.subplot(gs[0, 2:4])
# ax3 = plt.subplot(gs[0, 4:6])
# ax4 = plt.subplot(gs[1, 1:3])
# ax5 = plt.subplot(gs[1, 3:5])
# ax6 = plt.subplot(gs[1, -1])
# # ax1 = fig.add_subplot(gs[:2, 0])
# # ax2 = fig.add_subplot(gs[2:4, 0])
# # ax3 = fig.add_subplot(gs[3:4, 0:1])
# # ax4 = fig.add_subplot(gs[:2, 1])
# # ax5 = fig.add_subplot(gs[2:4, 1])
# # ax6 = fig.add_subplot(gs[5, 1])

train_fill_plot_list = [
    (mass, mass_range, ax1, "Mass", "Mass", "$(kg)$", (0, 0)),
    (ixx, ixx_range, ax2, "Inertia XX", "$I_{xx}$", "$(kg \ m^2)$", (-2, -2)),
    (iyy, iyy_range, ax3, "Inertia YY", "$I_{yy}$", "$(kg \ m^2)$", (-2, -2)),
    (izz, izz_range, ax4, "Inertia ZZ", "$I_{zz}$", "$(kg \ m^2)$", (-2, -2)),
    (
        km_kf,
        km_kf_range,
        ax5,
        "Propellor Constant",
        "$K_m/K_f$",
        "$(m)$",
        (0, 0),
    ),
]

for i, (data, data_range, ax, title, label, unit, scilimit) in enumerate(
    train_fill_plot_list
):
    ax.fill_between(
        arm_lengths,
        data_range[0],
        data_range[1],
        alpha=0.65,
        color="steelblue",
        label="Train Range",
    )
    ax.fill_between(
        arm_lengths,
        data_range[0],
        data_range[1],
        alpha=0.35,
        color="steelblue",
        label="Evaluation Range",
        hatch="\\",
    )
    ax.plot(arm_lengths, data, color="royalblue")
    ax.grid()
    # ax.set_title(title)
    ax.set_aspect("auto")
    # ax.set_xlabel("Arm Length (m)")
    ax.set_ylabel(f"{label} {unit}", fontsize=10)
    ax.ticklabel_format(axis="y", style="sci", scilimits=scilimit)
    # ax.tick_params(axis="x", labelsize=8)
    ax.tick_params(axis="y", labelsize=8)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(4))

## --------------- EVAL RANGE PLOT ----------------- ##


_sc_lengths = cfg.environment.scale_lengths
arm_lengths = np.linspace(_sc_lengths[1], _sc_lengths[1] + 0.04, 100)

mass = np.polyval(cfg.scale.avg_mass_fit, arm_lengths)
mass_std = np.polyval(cfg.scale.std_mass_fit, arm_lengths)
mass_range = np.array([mass - mass_std, mass + mass_std])
mass_range[mass_range < 0] = 0

ixx = np.polyval(cfg.scale.avg_ixx_fit, arm_lengths)
ixx_std = np.polyval(cfg.scale.std_ixx_fit, arm_lengths)
ixx_range = np.array([ixx - ixx_std, ixx + ixx_std])
ixx_range[ixx_range < 0] = 0

iyy = np.polyval(cfg.scale.avg_iyy_fit, arm_lengths)
iyy_std = np.polyval(cfg.scale.std_iyy_fit, arm_lengths)
iyy_range = np.array([iyy - iyy_std, iyy + iyy_std])
iyy_range[iyy_range < 0] = 0

izz = np.polyval(cfg.scale.avg_izz_fit, arm_lengths)
izz_std = np.polyval(cfg.scale.std_izz_fit, arm_lengths)
izz_range = np.array([izz - izz_std, izz + izz_std])
izz_range[izz_range < 0] = 0

km_kf = np.polyval(cfg.scale.avg_km_kf_fit, arm_lengths)
km_kf_std = np.polyval(cfg.scale.std_km_kf_fit, arm_lengths)
km_kf_range = np.array([km_kf - km_kf_std, km_kf + km_kf_std])
km_kf_range[km_kf_range < 0] = 0


eval_fill_plot_list = [
    (mass, mass_range, ax1, "Mass", "Mass", "(kg)"),
    (ixx, ixx_range, ax2, "Inertia XX", "I_xx", "(kg m^2)"),
    (iyy, iyy_range, ax3, "Inertia YY", "I_yy", "(kg m^2)"),
    (izz, izz_range, ax4, "Inertia ZZ", "I_zz", "(kg m^2)"),
    (km_kf, km_kf_range, ax5, "Propellor Constant", "K_m/K_f", "(m)"),
]

for i, (data, data_range, ax, title, label, unit) in enumerate(eval_fill_plot_list):
    ax.fill_between(
        arm_lengths,
        data_range[0],
        data_range[1],
        alpha=0.5,
        color="steelblue",
        hatch="X",
    )
    ax.plot(arm_lengths, data, color="royalblue")
    # ax.tick_params(axis="x", labelsize=8)

# ax5.tick_params(axis="x", labelsize=8)
# ax5.set_xlabel("Arm Length (m)")
# ax6.axis("off")
legend = ax5.get_legend_handles_labels()
# ax6.legend(
#     legend[0],
#     legend[1],
#     loc="center",
#     fontsize=9,
#     bbox_to_anchor=(0.5, 0.5),
#     fancybox=True,
#     shadow=True,
#     ncols=2,
# )

plt.legend(
    legend[0],
    legend[1],
    loc="upper center",
    fontsize=9,
    fancybox=True,
    shadow=True,
    # orientation="horizontal",
    bbox_to_anchor=(0.5, -0.3),
    ncols=2,
)

# plt.tight_layout()
plt.show()
