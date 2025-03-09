import numpy as np
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-deep")

# latex settings for matplotlib
from dataclasses import dataclass
import tyro
from adapt_drones.cfgs.config import *


@dataclass
class Args:
    env_id: str = "traj_v3"
    run_name: str = "sweet-feather-28"
    seed: int = 20240915
    agent: str = "RMA_DATT"
    scale: bool = True
    wind_bool: bool = True


args = tyro.cli(Args)

cfg = Config(
    env_id=args.env_id,
    seed=args.seed,
    eval=True,
    run_name=args.run_name,
    agent=args.agent,
    scale=args.scale,
)
run_folder = (
    "runs/"
    + cfg.experiment.wandb_project_name
    + "/"
    + cfg.grp_name
    + "/"
    + cfg.run_name
    + "/"
)
results_folder = run_folder + "results-icra/"
prefix = "wind_" if args.wind_bool else "nowind_"
phase_1_results = np.load(results_folder + prefix + "phase_1_scale.npy")
RMA_DATT_results = np.load(results_folder + prefix + "phase_1_scale.npy")

idx_sort_phase = np.argsort(np.mean(phase_1_results[:, :, 2], axis=1))
sorted_phase_1 = phase_1_results[idx_sort_phase]

phase_1 = sorted_phase_1[5:-5]  # remove the top 5 and bottom 5 seeds


mosaic = [["main"], ["mass"], ["ixx"], ["iyy"], ["izz"]]
fig, axs = plt.subplot_mosaic(
    mosaic,
    figsize=(6, 12),
    height_ratios=[4, 2, 2, 2, 2],
    sharex=True,
    constrained_layout=True,
)
arm_length = phase_1_results[0, :, 0]

mean_error = np.mean(phase_1[:, :, 2], axis=0)
std_mean = np.std(phase_1[:, :, 2], axis=0)

rms_error = np.mean(phase_1[:, :, 3], axis=0)
std_rms = np.std(phase_1[:, :, 3], axis=0)

axs["main"].errorbar(
    arm_length,
    mean_error,
    yerr=std_mean,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2,
    capthick=1,
)
# axs["main"].plot(arm_length, mean_error, label="Mean Error", color="firebrick")
# axs["main"].errorbar(
#     arm_length,
#     rms_error,
#     yerr=std_rms,
#     fmt=".",
#     color="steelblue",
#     ecolor="lightskyblue",
#     capsize=2,
#     capthick=1,
# )
# axs["main"].set_xlabel("Arm Length")
axs["main"].set_ylabel("Mean Position Error (m)")
axs["main"].grid()

mean_mass = np.mean(phase_1[:, :, 4], axis=0)
std_mass = np.std(phase_1[:, :, 4], axis=0)
axs["mass"].errorbar(
    arm_length,
    mean_mass,
    yerr=std_mass,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2.5,
    capthick=1,
)
axs["mass"].set_ylabel("Mass (kg)")
axs["mass"].grid()

mean_ixx = np.mean(phase_1[:, :, 5], axis=0)
std_ixx = np.std(phase_1[:, :, 5], axis=0)
axs["ixx"].errorbar(
    arm_length,
    mean_ixx,
    yerr=std_ixx,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2.5,
    capthick=1,
)
axs["ixx"].set_ylabel("Ixx (kg m^2)")
axs["ixx"].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
axs["ixx"].grid()

mean_iyy = np.mean(phase_1[:, :, 6], axis=0)
std_iyy = np.std(phase_1[:, :, 6], axis=0)
axs["iyy"].errorbar(
    arm_length,
    mean_iyy,
    yerr=std_iyy,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2.5,
    capthick=1,
)
axs["iyy"].set_ylabel("Iyy (kg m^2)")
axs["iyy"].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
axs["iyy"].grid()

mean_izz = np.mean(phase_1[:, :, 7], axis=0)
std_izz = np.std(phase_1[:, :, 7], axis=0)
axs["izz"].errorbar(
    arm_length,
    mean_izz,
    yerr=std_izz,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2.5,
    capthick=1,
)
axs["izz"].set_ylabel("Izz (kg m^2)")
axs["izz"].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
axs["izz"].grid()
plt.xlabel("Arm Length (m)")

plt.savefig(results_folder + "phase_1_plot.png")

# plt.show()

############## do the same for RMA_DATT ####################
idx_sort_rma = np.argsort(np.mean(RMA_DATT_results[:, :, 2], axis=1))
sorted_rma = RMA_DATT_results[idx_sort_rma]

rma_datt = sorted_rma[:10]

mosaic = [["main"], ["mass"], ["ixx"], ["iyy"], ["izz"]]
fig, axs = plt.subplot_mosaic(
    mosaic,
    figsize=(6, 12),
    height_ratios=[4, 2, 2, 2, 2],
    sharex=True,
    constrained_layout=True,
)
arm_length = RMA_DATT_results[0, :, 0]

mean_error = np.mean(rma_datt[:, :, 2], axis=0)
std_mean = np.std(rma_datt[:, :, 2], axis=0)

rms_error = np.mean(rma_datt[:, :, 3], axis=0)
std_rms = np.std(rma_datt[:, :, 3], axis=0)

axs["main"].errorbar(
    arm_length,
    mean_error,
    yerr=std_mean,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2,
    capthick=1,
)
# axs["main"].plot(arm_length, mean_error, label="Mean Error", color="firebrick")
# axs["main"].errorbar(
#     arm_length,
#     rms_error,
#     yerr=std_rms,
#     fmt=".",
#     color="steelblue",
#     ecolor="lightskyblue",
#     capsize=2,
#     capthick=1,
# )
# axs["main"].set_xlabel("Arm Length")
axs["main"].set_ylabel("Mean Position Error (m)")
axs["main"].grid()

mean_mass = np.mean(rma_datt[:, :, 4], axis=0)
std_mass = np.std(rma_datt[:, :, 4], axis=0)
axs["mass"].errorbar(
    arm_length,
    mean_mass,
    yerr=std_mass,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2.5,
    capthick=1,
)
axs["mass"].set_ylabel("Mass (kg)")
axs["mass"].grid()

mean_ixx = np.mean(rma_datt[:, :, 5], axis=0)
std_ixx = np.std(rma_datt[:, :, 5], axis=0)
axs["ixx"].errorbar(
    arm_length,
    mean_ixx,
    yerr=std_ixx,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2.5,
    capthick=1,
)
axs["ixx"].set_ylabel("Ixx (kg m^2)")
axs["ixx"].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
axs["ixx"].grid()

mean_iyy = np.mean(rma_datt[:, :, 6], axis=0)
std_iyy = np.std(rma_datt[:, :, 6], axis=0)
axs["iyy"].errorbar(
    arm_length,
    mean_iyy,
    yerr=std_iyy,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2.5,
    capthick=1,
)
axs["iyy"].set_ylabel("Iyy (kg m^2)")
axs["iyy"].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
axs["iyy"].grid()

mean_izz = np.mean(rma_datt[:, :, 7], axis=0)
std_izz = np.std(rma_datt[:, :, 7], axis=0)
axs["izz"].errorbar(
    arm_length,
    mean_izz,
    yerr=std_izz,
    fmt=".",
    color="firebrick",
    ecolor="lightcoral",
    capsize=2.5,
    capthick=1,
)
axs["izz"].set_ylabel("Izz (kg m^2)")
axs["izz"].ticklabel_format(axis="y", style="sci", scilimits=(-2, -2))
axs["izz"].grid()

plt.xlabel("Arm Length (m)")

plt.savefig(results_folder + "rma_datt_scale.png")
