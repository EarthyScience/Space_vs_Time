"""
Produce figure 2: Model Performance
"""

#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.dates as mdates

from mpl_toolkits.axes_grid1.inset_locator import inset_axes

#%%

# load csv

csv_path = f"../trained_models/version1/results/model_performance"

basic_obs = pd.read_csv(f"{csv_path}/basic_obs_perfomance.csv", header=0, index_col=0)
time_set= pd.read_csv(f"{csv_path}/time_perfomance.csv", header=0, index_col=0)
space_set = pd.read_csv(f"{csv_path}/space_perfomance.csv", header=0, index_col=0)
spacetime_set = pd.read_csv(f"{csv_path}/timespace_perfomance.csv", header=0, index_col=0)

# plt.plot r2_toal_time

#%%
# Set plot style
plt.rcParams["font.family"] = "TeX Gyre PagellaX"
plt.rcParams["font.size"] = 11
plt.rcParams['xtick.major.size'] = 5  # X-axis major ticks
plt.rcParams['ytick.major.size'] = 5

# set plot size

width_in = 120 / 25.4
height_in = 100 / 25.4

# %%

time_r2 = np.concatenate([basic_obs["r2_total"].values, time_set["r2_total"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_total"].values, space_set["r2_total"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_total"].values, spacetime_set["r2_total"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_total_std"].values, time_set["r2_total_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_total_std"].values, space_set["r2_total_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_total_std"].values, spacetime_set["r2_total_std"].iloc[:6].values])

step = [1,1.6,2.6,3.6,4.6,5.6,6.6]

fig = plt.figure(figsize=(width_in, height_in))
ax = fig.subplots()


# Plot with error bars
ax.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")
ax.scatter(1, 0.559, marker="*", color="#005555", s=95, label="Observation based")

ax.legend(loc=0, frameon=False)
ax.set_xticks(step, [80,200,400,600,800,1000,1200])
ax.set_xlabel("Number of Training Samples")
ax.set_ylabel("$R^2$")
#ax.set_title("Total $R^2$ for 30 test sites (30 years per site)")
ax.set_ylim(0,1)
fig.tight_layout()
fig.savefig(f"{csv_path}/total_r2_comp.png", dpi=300)
# %%

time_r2 = np.concatenate([basic_obs["r2_anom"].values, time_set["r2_anom"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_anom"].values, space_set["r2_anom"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_anom"].values, spacetime_set["r2_anom"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, time_set["r2_anom_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, space_set["r2_anom_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, spacetime_set["r2_anom_std"].iloc[:6].values])



fig = plt.figure(figsize=(width_in, height_in))
ax = fig.subplots()

# Plot with error bars
ax.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")

ax.legend(loc=0, frameon=False)
ax.set_xticks(step, [80,200,400,600,800,1000,1200])
ax.set_xlabel("Number of Training Samples")
ax.set_ylabel("$R^2_{anom}$")
#ax.set_title("$R^2$ anomalies for 30 test sites (30 years per site)")
ax.set_ylim(-0.05,1)
fig.tight_layout()
fig.savefig(f"{csv_path}/anom_r2_comp.png", dpi=300)
# %%

time_r2 = np.concatenate([basic_obs["r2_iav"].values, time_set["r2_iav"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_iav"].values, space_set["r2_iav"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_iav"].values, spacetime_set["r2_iav"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, time_set["r2_iav_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, space_set["r2_iav_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, spacetime_set["r2_iav_std"].iloc[:6].values])



fig = plt.figure(figsize=(width_in, height_in))
ax = fig.subplots()

# Plot with error bars
ax.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")
ax.scatter(1, -1.035, marker="*", color="#005555", s=95, label="Observation based")

ax.legend(loc=0, frameon=False)
ax.set_xticks(step, [80,200,400,600,800,1000,1200])
ax.set_xlabel("Number of Training Samples")
ax.set_ylabel("$R^2_{IAV}$")
#ax.set_title("$R^2$ IAV for 30 test sites (30 years per site)")
ax.set_ylim(-2,1)
fig.tight_layout()
fig.savefig(f"{csv_path}/iav_r2_comp.png", dpi=300)
# %%

# load values for denormalisation
load_path_denorm = "../data/pre_processing_steps"
tar_max_denrom = np.load(f"{load_path_denorm}/tar_max_normval.npy")
tar_min_denorm = np.load(f"{load_path_denorm}/tar_min_normval.npy")

save_path = f"../trained_models/version1/ig_results_anom"

# denormalize data
def denormalize_minmax(y_norm, y_min, y_max):
    """
    Denormalizes data that was normalized using min-max scaling.

    Parameters:
    - y_norm: Normalized values (array or scalar)
    - y_min: Minimum of the original data
    - y_max: Maximum of the original data

    Returns:
    - Denormalized values
    """
    return (y_norm * (y_max - y_min) + y_min) * 0.2729 * 86400 * 1000

#%%

#############################
### create paper figure 2 ###

width_in = 170 / 25.4
height_in = 150 / 25.4

step = [1,1.6,2.6,3.6,4.6,5.6,6.6]

# Create figure
fig = plt.figure(figsize=(width_in, height_in))

# Create a GridSpec with 2 rows and 3 columns
gs = gridspec.GridSpec(2, 3, height_ratios=[1, 1])  # First row is taller

# Create the first subplot spanning all columns in the first row
ax0 = fig.add_subplot(gs[1, :])
ax1 = fig.add_subplot(gs[0,0])
ax2 = fig.add_subplot(gs[0,1])
ax3 = fig.add_subplot(gs[0,2])

# Define labels and corresponding axes
labels = ["A", "B", "C"]
axes = [ax1, ax2, ax3]

# Loop through each axis and add a large black letter
for ax, label in zip(axes, labels):
    ax.text(
        -0.1, 1.05,  # Position slightly outside the top-left corner
        label,
        transform=ax.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
        fontsize=16,  # Adjust size as needed
        fontweight="black",
        color="black"
    )

ax0.text(
    -0.05, 1.05,  # Position slightly outside the top-left corner
    "D",
    transform=ax0.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
    fontsize=16,  # Adjust size as needed
    fontweight="black",
    color="black"
)

##########################
### plot r2 on b) ax01 ###

time_r2 = np.concatenate([basic_obs["r2_total"].values, time_set["r2_total"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_total"].values, space_set["r2_total"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_total"].values, spacetime_set["r2_total"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_total_std"].values, time_set["r2_total_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_total_std"].values, space_set["r2_total_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_total_std"].values, spacetime_set["r2_total_std"].iloc[:6].values])

# Plot with error bars
ax1.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax1.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax1.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")
#ax1.scatter(1, 0.559, marker="*", color="#005555", s=95, label="Obs based")

ax1.set_xticks(step, [80,200,400,600,800,1000,1200], fontsize=10)
ax1.tick_params(axis="x", rotation=-45)
#ax1.set_xlabel("Number of Training Samples")
ax1.set_title("$R^2$")
#ax.set_title("Total $R^2$ for 30 test sites (30 years per site)")
ax1.set_ylim(0.35,0.82)
#ax1.legend(loc="lower right", frameon=False, fontsize=8.5)

############################
### plot IAV on c) ax02 ###

time_r2 = np.concatenate([basic_obs["r2_iav"].values, time_set["r2_iav"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_iav"].values, space_set["r2_iav"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_iav"].values, spacetime_set["r2_iav"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, time_set["r2_iav_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, space_set["r2_iav_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, spacetime_set["r2_iav_std"].iloc[:6].values])

# Plot with error bars
ax2.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax2.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax2.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")
#ax2.scatter(1, -1.035, marker="*", color="#005555", s=95, label="Observation based")

#ax[1,0].legend(loc=0, frameon=False)
ax2.set_xticks(step, [80,200,400,600,800,1000,1200], fontsize=10)
ax2.tick_params(axis="x", rotation=-45)
ax2.set_xlabel("Number of Training Samples")
ax2.set_ylim(-1.2,0.55)
ax2.set_title("$R^2_{IAV}$")

###############################
### plot r2 anom on d) ax03 ###

time_r2 = np.concatenate([basic_obs["r2_anom"].values, time_set["r2_anom"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_anom"].values, space_set["r2_anom"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_anom"].values, spacetime_set["r2_anom"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, time_set["r2_anom_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, space_set["r2_anom_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, spacetime_set["r2_anom_std"].iloc[:6].values])


# Plot with error bars
ax3.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax3.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax3.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")

#ax.legend(loc=0, frameon=False)
ax3.set_xticks(step, [80,200,400,600,800,1000,1200], fontsize=10)
ax3.tick_params(axis="x", rotation=-45)
#ax3.set_xlabel("Number of Training Samples")
ax3.set_title("$R^2_{anom}$")
#ax.set_title("$R^2$ anomalies for 30 test sites (30 years per site)")
ax3.set_ylim(-0.05,0.42)
ax3.legend(loc="lower right", bbox_to_anchor=(1.03, -0.04), frameon=False, fontsize=9)


##############################
### plot timeseries on ax0 ###

site_index = 5
year_index = 23

# path
path = f"../trained_models/version1/results"

# load data
target = np.load(f"{path}/nep_target_siteyear.npy")
space = np.load(f"{path}/nep_model_space_siteyear.npy")
time = np.load(f"{path}/nep_model_time_siteyear.npy")
time_space = np.load(f"{path}/nep_model_timespace_siteyear.npy")

# extract years -1, +1
num_years = 1

target_sample = target[site_index, year_index:year_index+1, 0,:].reshape((num_years*365))
space_sample = space[site_index, year_index:year_index+1, 0,:].reshape((num_years*365))
time_sample = time[site_index, year_index:year_index+1, 0,:].reshape((num_years*365))
time_space_sample = time_space[site_index, year_index:year_index+1, 0,:].reshape((num_years*365))

# denormalize data


target_sample_denorm = denormalize_minmax(target_sample, tar_min_denorm[0], tar_max_denrom[0])
space_sample_denorm = denormalize_minmax(space_sample, tar_min_denorm[0], tar_max_denrom[0])
time_sample_denorm = denormalize_minmax(time_sample, tar_min_denorm[0], tar_max_denrom[0])
time_space_sample_denorm = denormalize_minmax(time_space_sample, tar_min_denorm[0], tar_max_denrom[0])


# Example: Create a time series from two years
dates = pd.date_range(start="2023-01-01", periods=365, freq="D")  # 2 years
#dates = np.arange(0,365*num_years)

ax0.plot(dates, target_sample_denorm, label="Target", color="#006c66")
ax0.plot(dates, space_sample_denorm, label="Space", color="#c6d325")
ax0.plot(dates, time_sample_denorm, label="Time", color="#ef7c00")
ax0.plot(dates, time_space_sample_denorm, label="Time + Space", color="#00b1ea")

ax0.set_ylabel("NEP (gC m$^{-2}$ d$^{-1}$)")
ax0.set_xlabel("Time (months)")

ax0.set_xlim(pd.Timestamp("2023-01-01"), pd.Timestamp("2024-01-01"))
ax0.xaxis.set_major_locator(mdates.MonthLocator(bymonth=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 1]))  # Every 2nd month
ax0.xaxis.set_major_formatter(mdates.DateFormatter("%b"))  # "Jan", "Mar", etc.
ax0.legend(loc=0, frameon=False, fontsize=11, handlelength=1.5)

fig.tight_layout()

##################################
### plot NEP trend inside plot ###

# Define the inset axis (relative position in ax2)
ax_inset = inset_axes(ax1, width="35%", height="30%", loc="lower right")  

# denorm trend
max_min_diff =  tar_max_denrom[0] - tar_min_denorm[0]

# Small scatter plot with error bars
tar_trend = np.array([time_set["trend_tar"].iloc[5], time_set["trend_tar_std"].iloc[5]]) * max_min_diff * 2.355*1e7 
time_trend = np.array([time_set["trend_mout"].iloc[5], time_set["trend_mout_std"].iloc[5]]) * max_min_diff * 2.355*1e7 
space_trend = np.array([space_set["trend_mout"].iloc[5], space_set["trend_mout_std"].iloc[5]]) * max_min_diff * 2.355*1e7 
timespace_trend = np.array([spacetime_set["trend_mout"].iloc[5], spacetime_set["trend_mout_std"].iloc[5]]) * max_min_diff * 2.355*1e7 

# denormalise
ax_inset.errorbar(1, tar_trend[0], yerr=tar_trend[1], marker="o", linestyle="", capsize=3, color="#006c66")
ax_inset.errorbar(2, space_trend[0], yerr=space_trend[1], marker="o", linestyle="", capsize=3, color="#c6d325")
ax_inset.errorbar(3, time_trend[0], yerr=time_trend[1], marker="o", linestyle="", capsize=3, color="#ef7c00")
ax_inset.errorbar(4, timespace_trend[0], yerr=timespace_trend[1], marker="o", linestyle="", capsize=3, color="#00b1ea")
ax_inset.set_xlim(0,5)
#ax_inset.set_ylabel("gC m$^{-2}$ y$^{-1}$ y$^{-1}$", fontsize=7)
ax_inset.tick_params(axis="both", labelsize=7)
ax_inset.yaxis.get_offset_text().set_fontsize(7)
ax_inset.set_title("NEP Trend", fontsize=8)

# Formatting inset
ax_inset.set_xticks([])
#ax_inset.set_yticks([])
ax_inset.spines["top"].set_visible(False)
ax_inset.spines["right"].set_visible(False)


fig.savefig(f"{csv_path}/figure2.png", dpi=600)
# %%

###################
### Plot für PP ###
###################

# Set plot style
plt.rcParams["font.family"] = "TeX Gyre PagellaX"
plt.rcParams["font.size"] = 12
plt.rcParams['xtick.major.size'] = 5  # X-axis major ticks
plt.rcParams['ytick.major.size'] = 5


width_in = 220 / 25.4
height_in = 90 / 25.4

step = [1,1.6,2.6,3.6,4.6,5.6,6.6]

# Create figure
fig = plt.figure(figsize=(width_in, height_in))


# Create the first subplot spanning all columns in the first row

ax1 = fig.add_subplot(131)
ax2 = fig.add_subplot(132)
ax3 = fig.add_subplot(133)

# Define labels and corresponding axes
labels = ["A", "B", "C"]
axes = [ax1, ax2, ax3]

# Loop through each axis and add a large black letter
for ax, label in zip(axes, labels):
    ax.text(
        -0.1, 1.05,  # Position slightly outside the top-left corner
        label,
        transform=ax.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
        fontsize=16,  # Adjust size as needed
        fontweight="black",
        color="black"
    )

##########################
### plot r2 on b) ax01 ###

time_r2 = np.concatenate([basic_obs["r2_total"].values, time_set["r2_total"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_total"].values, space_set["r2_total"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_total"].values, spacetime_set["r2_total"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_total_std"].values, time_set["r2_total_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_total_std"].values, space_set["r2_total_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_total_std"].values, spacetime_set["r2_total_std"].iloc[:6].values])

# Plot with error bars
ax1.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax1.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax1.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")
#ax1.scatter(1, 0.559, marker="*", color="#005555", s=95, label="Obs based")

ax1.set_xticks(step, [80,200,400,600,800,1000,1200], fontsize=11)
ax1.tick_params(axis="x", rotation=-45)
#ax1.set_xlabel("Number of Training Samples")
ax1.set_title("$R^2$")
#ax.set_title("Total $R^2$ for 30 test sites (30 years per site)")
ax1.set_ylim(0.35,0.82)
#ax1.legend(loc="lower right", frameon=False, fontsize=8.5)

############################
### plot IAV on c) ax02 ###

time_r2 = np.concatenate([basic_obs["r2_iav"].values, time_set["r2_iav"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_iav"].values, space_set["r2_iav"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_iav"].values, spacetime_set["r2_iav"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, time_set["r2_iav_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, space_set["r2_iav_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_iav_std"].values, spacetime_set["r2_iav_std"].iloc[:6].values])

# Plot with error bars
ax2.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax2.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax2.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")
#ax2.scatter(1, -1.035, marker="*", color="#005555", s=95, label="Observation based")

#ax[1,0].legend(loc=0, frameon=False)
ax2.set_xticks(step, [80,200,400,600,800,1000,1200], fontsize=11)
ax2.tick_params(axis="x", rotation=-45)
ax2.set_xlabel("Number of Training Samples")
ax2.set_ylim(-1.2,0.55)
ax2.set_title("$R^2_{IAV}$")

###############################
### plot r2 anom on d) ax03 ###

time_r2 = np.concatenate([basic_obs["r2_anom"].values, time_set["r2_anom"].iloc[:6].values])
space_r2 = np.concatenate([basic_obs["r2_anom"].values, space_set["r2_anom"].iloc[:6].values])
timespace_r2 = np.concatenate([basic_obs["r2_anom"].values, spacetime_set["r2_anom"].iloc[:6].values])

time_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, time_set["r2_anom_std"].iloc[:6].values])
space_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, space_set["r2_anom_std"].iloc[:6].values])
timespace_r2_std = np.concatenate([basic_obs["r2_anom_std"].values, spacetime_set["r2_anom_std"].iloc[:6].values])


# Plot with error bars
ax3.errorbar(step, time_r2, yerr=time_r2_std, label="Time", marker="o", capsize=5, color="#ef7c00")
ax3.errorbar(step, space_r2, yerr=space_r2_std, label="Space", marker="o", capsize=5, color="#c6d325")
ax3.errorbar(step, timespace_r2, yerr=timespace_r2_std, label="Time + Space", marker="o", capsize=5, color="#00b1ea")
ax3.scatter(1,-10, color="#006c66", label="Target")

#ax.legend(loc=0, frameon=False)
ax3.set_xticks(step, [80,200,400,600,800,1000,1200], fontsize=11)
ax3.tick_params(axis="x", rotation=-45)
#ax3.set_xlabel("Number of Training Samples")
ax3.set_title("$R^2_{anom}$")
#ax.set_title("$R^2$ anomalies for 30 test sites (30 years per site)")
ax3.set_ylim(-0.05,0.42)
ax3.legend(loc="lower right", frameon=False, fontsize=10)

fig.tight_layout()

##################################
### plot NEP trend inside plot ###

# Define the inset axis (relative position in ax2)
ax_inset = inset_axes(ax1, width="35%", height="30%", loc="lower right")  

# denorm trend
max_min_diff =  tar_max_denrom[0] - tar_min_denorm[0]

# Small scatter plot with error bars
tar_trend = np.array([time_set["trend_tar"].iloc[5], time_set["trend_tar_std"].iloc[5]]) * max_min_diff * 2.355*1e7 
time_trend = np.array([time_set["trend_mout"].iloc[5], time_set["trend_mout_std"].iloc[5]]) * max_min_diff * 2.355*1e7 
space_trend = np.array([space_set["trend_mout"].iloc[5], space_set["trend_mout_std"].iloc[5]]) * max_min_diff * 2.355*1e7 
timespace_trend = np.array([spacetime_set["trend_mout"].iloc[5], spacetime_set["trend_mout_std"].iloc[5]]) * max_min_diff * 2.355*1e7 

# denormalise
ax_inset.errorbar(1, tar_trend[0], yerr=tar_trend[1], marker="o", linestyle="", capsize=3, color="#006c66")
ax_inset.errorbar(2, space_trend[0], yerr=space_trend[1], marker="o", linestyle="", capsize=3, color="#c6d325")
ax_inset.errorbar(3, time_trend[0], yerr=time_trend[1], marker="o", linestyle="", capsize=3, color="#ef7c00")
ax_inset.errorbar(4, timespace_trend[0], yerr=timespace_trend[1], marker="o", linestyle="", capsize=3, color="#00b1ea")
ax_inset.set_xlim(0,5)
#ax_inset.set_ylabel("gC m$^{-2}$ y$^{-1}$ y$^{-1}$", fontsize=7)
ax_inset.tick_params(axis="both", labelsize=9)
ax_inset.yaxis.get_offset_text().set_fontsize(9)
ax_inset.set_title("NEP Trend", fontsize=10)

# Formatting inset
ax_inset.set_xticks([])
#ax_inset.set_yticks([])
ax_inset.spines["top"].set_visible(False)
ax_inset.spines["right"].set_visible(False)


fig.savefig(f"{csv_path}/figure2_pp.png", dpi=600)
# %%
