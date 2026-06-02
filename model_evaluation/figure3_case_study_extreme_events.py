"""
Produce figure 3:

1. Extract two years (low NEP uptake, high NEP uptake)
2. Plot both years
3. Apply IG on specific time window to derive the model importance
4. Plot the data for energy and water

"""

# %%
import torch
import pywt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import matplotlib.gridspec as gridspec

from ml_models import Model18_IG_sum_casestudy
from captum.attr import IntegratedGradients

from matplotlib import colors as mcolors

from pathlib import Path
import os

# %%

# path
path = f"../trained_models/version1/results"

# load data
target = np.load(f"{path}/nep_target_siteyear.npy")
space = np.load(f"{path}/nep_model_space_siteyear.npy")
time = np.load(f"{path}/nep_model_time_siteyear.npy")
time_space = np.load(f"{path}/nep_model_timespace_siteyear.npy")

save_path = "../trained_models/version1/ig_results/plots"


# %%

## calculate log scales for IG plots

# Plot
def gen_log_space(
    limit, n
):  # from https://stackoverflow.com/questions/12418234/logarithmically-spaced-integers
    result = [1]
    ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    while len(result) < n:
        next_value = result[-1] * ratio
        if next_value - result[-1] >= 1:
            # safe zone. next_value will be a different integer
            result.append(next_value)
        else:
            # problem! same integer. we need to find next_value by artificially incrementing previous value
            result.append(result[-1] + 1)
            # recalculate the ratio so that the remaining values will scale correctly
            ratio = (float(limit) / result[-1]) ** (1.0 / (n - len(result)))
    # round, re-adjust to 0 indexing (i.e. minus 1) and return np.uint64 array
    return np.array(list(map(lambda x: round(x) - 1, result)), dtype=np.uint64)


scales = gen_log_space(731, 64 + 2)
scales_logsp = scales[2:] / 4
periods = 1 / pywt.scale2frequency("mexh", scales_logsp)


# %%
#######################
### denormalisation ###

# Load values for denormalization
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

load_path_denorm = DATA_ROOT / "pre_processing_steps"

tar_max_denorm = np.load(load_path_denorm / "tar_max_normval.npy")
tar_min_denorm = np.load(load_path_denorm / "tar_min_normval.npy")

feat_max_denorm = np.load(
    load_path_denorm / "feat_max_normval.npy"
).squeeze()

feat_min_denorm = np.load(
    load_path_denorm / "feat_min_normval.npy"
).squeeze()

save_path = f"../trained_models/version1/ig_results_anom"


# denormalize data
def denormalize_minmax(y_norm, y_min, y_max):
    """
    Denormalizes data that was normalized using min-max scaling and convert to correct unit.

    Parameters:
    - y_norm: Normalized values (array or scalar)
    - y_min: Minimum of the original data
    - y_max: Maximum of the original data

    Returns:
    - Denormalized values
    """
    return (y_norm * (y_max - y_min) + y_min) * 0.2729 * 86400 * 1000


# %%
site_index_low = 5
year_index_low = 23

# extract years -1, +1
num_years = 1

low_target_sample = target[
    site_index_low, year_index_low : year_index_low + 1, 0, :
].reshape((num_years * 365))
low_space_sample = space[
    site_index_low, year_index_low : year_index_low + 1, 0, :
].reshape((num_years * 365))
low_time_sample = time[
    site_index_low, year_index_low : year_index_low + 1, 0, :
].reshape((num_years * 365))
low_time_space_sample = time_space[
    site_index_low, year_index_low : year_index_low + 1, 0, :
].reshape((num_years * 365))

low_target_sample_denorm = denormalize_minmax(
    low_target_sample, tar_min_denorm[0], tar_max_denorm[0]
)
low_space_sample_denorm = denormalize_minmax(
    low_space_sample, tar_min_denorm[0], tar_max_denorm[0]
)
low_time_sample_denorm = denormalize_minmax(
    low_time_sample, tar_min_denorm[0], tar_max_denorm[0]
)
low_time_space_sample_denorm = denormalize_minmax(
    low_time_space_sample, tar_min_denorm[0], tar_max_denorm[0]
)

dates = np.arange(0, 365 * num_years)

width_in = 200 / 25.4  # 130 / 25.4
height_in = 100 / 25.4

fig = plt.figure(figsize=(width_in, height_in))
ax = fig.add_subplot()

ax.plot(dates, low_target_sample, label="Target", color="#006c66")
ax.plot(dates, low_space_sample, label="Space", color="#c6d325")
ax.plot(dates, low_time_sample, label="Time", color="#ef7c00")
ax.plot(dates, low_time_space_sample, label="Time Space", color="#00b1ea")

start_index_low = 210
end_index_low = 250

ax.vlines(start_index_low, 0.2, 0.8, colors="black", linestyles="--")
ax.vlines(end_index_low, 0.2, 0.8, colors="black", linestyles="--")

# %%
# extract sample with high carbon uptake
load_path = f"../trained_models/version1/ig_results_anom"

index_high_nep_years = pd.read_csv(f"{load_path}, anom_99_years_30.csv")


def convert_flux_kgCO2_to_gC_per_m2_yr(flux_kgCO2_per_m2_s):
    """
    Converts flux from kg(CO2) / (m^2 * s) to gC / (m^2 * yr).

    Parameters:
    - flux_kgCO2_per_m2_s: NumPy array or scalar, flux in kg CO2 per (m^2 * s)

    Returns:
    - Flux in gC per (m^2 * yr)
    """
    conversion_factor = 2.355 * 1e7  # From kg(CO2)/(m^2*s) to gC/(m^2*yr)
    return flux_kgCO2_per_m2_s * conversion_factor


index_high_nep_years.loc[:, "Target Sum"] = convert_flux_kgCO2_to_gC_per_m2_yr(
    index_high_nep_years.loc[:, "Target Sum"]
)
index_high_nep_years.loc[:, "Space Sum"] = convert_flux_kgCO2_to_gC_per_m2_yr(
    index_high_nep_years.loc[:, "Space Sum"]
)
index_high_nep_years.loc[:, "Time Sum"] = convert_flux_kgCO2_to_gC_per_m2_yr(
    index_high_nep_years.loc[:, "Time Sum"]
)

site_index_high = 23
year_index_high = 27

# extract years -1, +1
num_years = 1

high_target_sample = target[
    site_index_high, year_index_high : year_index_high + 1, 0, :
].reshape((num_years * 365))
high_space_sample = space[
    site_index_high, year_index_high : year_index_high + 1, 0, :
].reshape((num_years * 365))
high_time_sample = time[
    site_index_high, year_index_high : year_index_high + 1, 0, :
].reshape((num_years * 365))
high_time_space_sample = time_space[
    site_index_high, year_index_high : year_index_high + 1, 0, :
].reshape((num_years * 365))

high_target_sample_denorm = denormalize_minmax(
    high_target_sample, tar_min_denorm[0], tar_max_denorm[0]
)
high_space_sample_denorm = denormalize_minmax(
    high_space_sample, tar_min_denorm[0], tar_max_denorm[0]
)
high_time_sample_denorm = denormalize_minmax(
    high_time_sample, tar_min_denorm[0], tar_max_denorm[0]
)
high_time_space_sample_denorm = denormalize_minmax(
    high_time_space_sample, tar_min_denorm[0], tar_max_denorm[0]
)


width_in = 200 / 25.4  # 130 / 25.4
height_in = 100 / 25.4

fig = plt.figure(figsize=(width_in, height_in))
ax = fig.add_subplot()

ax.plot(dates, high_target_sample, label="Target", color="#006c66")
ax.plot(dates, high_space_sample, label="Space", color="#c6d325")
ax.plot(dates, high_time_sample, label="Time", color="#ef7c00")
ax.plot(dates, high_time_space_sample, label="Time Space", color="#00b1ea")

start_index_high = 194 # 90
end_index_high = 299 # 145

ax.vlines(start_index_high, 0.5, 0.8, colors="black", linestyles="--")
ax.vlines(end_index_high, 0.5, 0.8, colors="black", linestyles="--")

#%%
# run IG on manipulated model

# load test data
device = torch.device("cpu")
test_data_path = (
    "/Net/Groups/BGI/scratch/drachti/data/jsbach_space_time_splits/datasets/test_sets"
)
# total test
test_feat = torch.from_numpy(np.load(f"{test_data_path}/test2_test_feat.npy")).float()
test_stat = torch.from_numpy(np.load(f"{test_data_path}/test2_test_stat.npy")).float()
# extract the last 30 years
test_feat = test_feat[:, -30:, :].to(device)
test_stat = test_stat[:, -30:, :].to(device)

# calculate IG for the specific anomaly years for both models
num_years = 30
# load the models

n_epochs = 300
# init model
device = torch.device("cpu")
num_input = 11
num_output = 365 * 2

start_index_low = int(
    start_index_low
)  # torch.tensor(start_index_low, dtype=torch.float64)
end_index_low = int(end_index_low)  # torch.tensor(end_index_low, dtype=torch.float64)

start_index_high = int(
    start_index_high
)  # torch.tensor(start_index_low, dtype=torch.float64)
end_index_high = int(end_index_high)  # torch.tensor(end_index_low, dtype=torch.float64)


model_space_low = Model18_IG_sum_casestudy(
    num_input, num_output, start_index_low, end_index_low
).to(device)
model_space_high = Model18_IG_sum_casestudy(
    num_input, num_output, start_index_high, end_index_high
).to(device)

model_time_low = Model18_IG_sum_casestudy(
    num_input, num_output, start_index_low, end_index_low
).to(device)
model_time_high = Model18_IG_sum_casestudy(
    num_input, num_output, start_index_high, end_index_high
).to(device)


model_set = "space"
model_path = (
    f"/Net/Groups/BGI/scratch/drachti/waveflux-space-or-time/trained_models/version1"
)


def calc_ig(ig, sitey_feat, sitey_stat, site_feat_mean, site_stat):
    """
    Calculate IG
    """

    attribution = ig.attribute(
        inputs=(sitey_feat, sitey_stat), target=0, baselines=(site_feat_mean, site_stat)
    )

    sample_ig = attribution[0].cpu().detach().numpy()

    # normalize to 1 # output shape: 11,64,730
    # sum up
    # sample_ig_sum = sample_ig.sum()

    # # Avoid division by zero
    # if sample_ig_sum == 0:
    #     print("Ig sum is zero")
    #     return sample_ig  # or handle it differently (e.g., return zeros)

    # # divide by sum
    # sample_ig = sample_ig / sample_ig_sum

    return sample_ig


# create numpy array for IG results
ig_space = np.zeros((5, 2, 11, 64, 730))
ig_time = np.zeros((5, 2, 11, 64, 730))

# loop over the model initalisations
for i in range(5):
    print(i)
    # load space model
    model_set = "space"
    model_code = f"resnet18_{model_set}_1"
    modelinit_path = (
        f"{model_path}/space_set/6_{model_code}_{n_epochs}/init_{i}/m{model_code}.pt"
    )
    model_space_low.load_state_dict(
        torch.load(modelinit_path, map_location=torch.device("cpu"))
    )
    model_space_high.load_state_dict(
        torch.load(modelinit_path, map_location=torch.device("cpu"))
    )

    # load time model
    model_set = "time"
    model_code = f"resnet18_{model_set}_1"
    modelinit_path = (
        f"{model_path}/time_set/6_{model_code}_{n_epochs}/init_{i}/m{model_code}.pt"
    )
    model_time_low.load_state_dict(
        torch.load(modelinit_path, map_location=torch.device("cpu"))
    )
    model_time_high.load_state_dict(
        torch.load(modelinit_path, map_location=torch.device("cpu"))
    )

    igmod_space_low = IntegratedGradients(model_space_low)
    igmod_space_high = IntegratedGradients(model_space_high)
    igmod_time_low = IntegratedGradients(model_time_low)
    igmod_time_high = IntegratedGradients(model_time_high)

    # index_site = int(bottom_entries_df.loc[j,"site"])
    # index_year = int(bottom_entries_df.loc[j,"year"])

    sample_feat_low = test_feat[site_index_low, year_index_low, :]
    sample_feat_low = sample_feat_low[None, :]
    sample_stat_low = test_stat[site_index_low, year_index_low, :]
    sample_stat_low = sample_stat_low[None, :]

    # calculate baseline for IG
    site_mean_feat_low = test_feat[site_index_low, :].mean(axis=0)
    site_mean_feat_low = site_mean_feat_low[None, :]
    site_mean_stat_low = test_stat[site_index_low, :].mean(axis=0)
    site_mean_stat_low = site_mean_stat_low[None, :]

    # extract data for high year
    sample_feat_high = test_feat[site_index_high, year_index_high, :]
    sample_feat_high = sample_feat_high[None, :]
    sample_stat_high = test_stat[site_index_high, year_index_high, :]
    sample_stat_high = sample_stat_high[None, :]

    # calculate baseline for IG
    site_mean_feat_high = test_feat[site_index_high, :].mean(axis=0)
    site_mean_feat_high = site_mean_feat_high[None, :]
    site_mean_stat_high = test_stat[site_index_high, :].mean(axis=0)
    site_mean_stat_high = site_mean_stat_high[None, :]

    # calculate IG
    ig_space[i, 0, :] = calc_ig(
        igmod_space_low,
        sample_feat_low,
        sample_stat_low,
        site_mean_feat_low,
        site_mean_stat_low,
    )
    ig_time[i, 0, :] = calc_ig(
        igmod_time_low,
        sample_feat_low,
        sample_stat_low,
        site_mean_feat_low,
        site_mean_stat_low,
    )
    ig_space[i, 1, :] = calc_ig(
        igmod_space_high,
        sample_feat_high,
        sample_stat_high,
        site_mean_feat_high,
        site_mean_stat_high,
    )
    ig_time[i, 1, :] = calc_ig(
        igmod_time_high,
        sample_feat_high,
        sample_stat_high,
        site_mean_feat_high,
        site_mean_stat_high,
    )


# calculate mean over models
ig_space_mean = ig_space.mean(axis=0)
ig_time_mean = ig_time.mean(axis=0)

ig_space_mean_sum = ig_space_mean.sum(axis=(1,2,3))
ig_time_mean_sum = ig_time_mean.sum(axis=(1,2,3))
#%%

def denormalize_ig(y_norm, y_min, y_max, x_min, x_max):
    """
    Denormalize IG and convert to correct unit
    """
    ig_denorm = np.zeros((2,11,64,730))
    for i in range(11):
        ig_denorm[:,i,:] = y_norm[:,i,:] * ((y_max-y_min)/(x_max[i]-x_min[i])) * 0.2729 * 86400 * 1000
    return ig_denorm

tar_min_array = np.array(tar_min_denorm[0])#np.full((1, 1, 1, 1), tar_min_denorm[0])
tar_max_array = np.array(tar_max_denorm[0])#np.full((1, 1, 1, 1), tar_max_denrom[0])

# denormalize IG values
#ig_space_mean = denormalize_ig(ig_space_mean, tar_min_array, tar_max_array, feat_min_denorm, feat_max_denrom)
#ig_time_mean = denormalize_ig(ig_time_mean, tar_min_array, tar_max_array, feat_min_denorm, feat_max_denrom)

#ig_space_mean_sum_denorm = denormalize_minmax(ig_space_mean_sum, tar_min_array, tar_max_array)
#ig_time_mean_sum_denorm = denormalize_minmax(ig_time_mean_sum, tar_min_array, tar_max_array)
# save IG results to disk
# np.save(f"{save_path}/anom_{perc}_ig_space_mean_{num_years}.npy", ig_space_mean)
# np.save(f"{save_path}/anom_{perc}_ig_time_mean_{num_years}.npy", ig_time_mean)


# %%

## start creating the plot

plt.rcParams["font.family"] = "TeX Gyre PagellaX"
plt.rcParams["font.size"] = 11
plt.rcParams["xtick.major.size"] = 5  # X-axis major ticks
plt.rcParams["ytick.major.size"] = 5

# other plot parameters
alpha_vlines = 0.6
alpha_vlines2 = 0.9
colors = "BrBG"
levels = 8
time = np.arange(0, 730)

width_in = 160 / 25.4
height_in = 170 / 25.4

fig = plt.figure(figsize=(width_in, height_in), constrained_layout=True)
# Create a GridSpec with 2 rows and 3 columns
gs = gridspec.GridSpec(2, 2, height_ratios=[1, 1], figure=fig)  # First row is taller

# Create the first subplot spanning all columns in the first row
ax0 = fig.add_subplot(gs[0,0])
ax1 = fig.add_subplot(gs[0,1])
ax2 = fig.add_subplot(gs[1,0])
ax3 = fig.add_subplot(gs[1,1])

# set letters for subplots

labels = ["A", "B", "C", "D"]
axes = [ax0, ax1, ax2, ax3]

# Loop through each axis and add a large black letter
for ax, label in zip(axes, labels):
    ax.text(
        -0.2, 1.05,  # Position slightly outside the top-left corner
        label,
        transform=ax.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
        fontsize=16,  # Adjust size as needed
        fontweight="black",
        color="black"
    )

ax0.text(
    -0.02, 1.05,  # Position slightly outside the top-left corner
    "Lat 51.294, Lon 24.375, Yr 2015",
    transform=ax0.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
    fontsize=13,  # Adjust size as needed
    fontweight="black",
    color="black"
)

ax1.text(
    -0.05, 1.05,  # Position slightly outside the top-left corner
    "Lat 45.699, Lon 266.250, Yr 2019",
    transform=ax1.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
    fontsize=13,  # Adjust size as needed
    fontweight="black",
    color="black"
)
   
# place text over plot C and D
ax2.text(
    0.18, 1.05,  # Position slightly outside the top-left corner
    "Space minus Time",
    transform=ax2.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
    fontsize=14,  # Adjust size as needed
    fontweight="black",
    color="black"
)

ax3.text(
    0.18, 1.05,  # Position slightly outside the top-left corner
    "Space minus Time",
    transform=ax3.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
    fontsize=14,  # Adjust size as needed
    fontweight="black",
    color="black"
)

# ax0.text(
#     -0.05, 1.05,  # Position slightly outside the top-left corner
#     "A",
#     transform=ax0.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
#     fontsize=16,  # Adjust size as needed
#     fontweight="black",
#     color="black"
# )

## plot 1

dates = np.arange(0, 365)

ax0.plot(dates, low_target_sample_denorm, label="Target", color="#006c66")
ax0.plot(dates, low_space_sample_denorm, label="Space", color="#c6d325")
ax0.plot(dates, low_time_sample_denorm, label="Time", color="#ef7c00")
ax0.plot(dates, low_time_space_sample_denorm, label="Time + Space", color="#00b1ea")

ax0.vlines(start_index_low, -3, 1.3, colors="black", linestyles="--", alpha=alpha_vlines2)
ax0.vlines(end_index_low, -3, 1.3, colors="black", linestyles="--", alpha=alpha_vlines2)

ax0.set_ylabel("NEP (gC m$^{-2}$ d$^{-1}$)")
#ax0.set_xlabel("Time (m)")

# ax.set_xlim(dates[0], pd.to_datetime("2006-01-01"))
ax0.set_xlim(0, 365)
ax0.set_ylim(-3, 1.3)
month_starts = [
    1,
    61,
    122,
    183,
    245,
    306,
    365,
    # 365 + 60,
    # 365 + 121,
    # 365 + 182,
    # 365 + 244,
    # 365 + 305,
    # 729,
]
month_names = [
    "Jan",
    "Mar",
    "May",
    "Jul",
    "Sep",
    "Nov",
    "Jan",
    # "Mar",
    # "May",
    # "Jul",
    # "Sep",
    # "Nov",
    # "Jan",
]
ax0.set_xticks(month_starts)
ax0.set_xticklabels(month_names)
ax0.legend(loc=0, frameon=False, fontsize=9, handlelength=1.5)

ax1.plot(dates, high_target_sample_denorm, label="Target", color="#006c66")
ax1.plot(dates, high_space_sample_denorm, label="Space", color="#c6d325")
ax1.plot(dates, high_time_sample_denorm, label="Time", color="#ef7c00")
ax1.plot(dates, high_time_space_sample_denorm, label="Time + Space", color="#00b1ea")

ax1.vlines(start_index_high, -0.35, 1.3, colors="black", linestyles="--", alpha=alpha_vlines2)
ax1.vlines(end_index_high, -0.35, 1.3, colors="black", linestyles="--", alpha=alpha_vlines2)

ax1.set_ylabel("NEP (gC m$^{-2}$ d$^{-1}$)")
#ax1.set_xlabel("Time (m)")

# ax.set_xlim(dates[0], pd.to_datetime("2006-01-01"))
ax1.set_xlim(0, 365)
ax1.set_ylim(-0.35, 1.3)
month_starts = [
    1,
    61,
    122,
    183,
    245,
    306,
    365,
    # 365 + 60,
    # 365 + 121,
    # 365 + 182,
    # 365 + 244,
    # 365 + 305,
    # 729,
]
month_names = [
    "Jan",
    "Mar",
    "May",
    "Jul",
    "Sep",
    "Nov",
    "Jan",
    # "Mar",
    # "May",
    # "Jul",
    # "Sep",
    # "Nov",
    # "Jan",
]
ax1.set_xticks(month_starts)
ax1.set_xticklabels(month_names)
# ax[0,1].legend(loc=0, frameon=False, fontsize=9, handlelength=1.5)

##### Low sample plots

# set all values below the random value threshold to Zero

ig_space_mean_low = ig_space_mean[0, :]
ig_time_mean_low = ig_time_mean[0, :]

ig_space_mean_high = ig_space_mean[1, :]
ig_time_mean_high = ig_time_mean[1, :]

# filter low example
space_low_thres = np.percentile(np.abs(ig_space_mean[0, -1, :]), 99.9)
time_low_thres = np.percentile(np.abs(ig_time_mean[0, -1, :]), 99.9)

ig_space_mean_low_filt = np.where(
    (ig_space_mean_low > space_low_thres) | (ig_space_mean_low < -space_low_thres),
    ig_space_mean_low,
    0,
)

ig_time_mean_low_filt = np.where(
    (ig_time_mean_low > time_low_thres) | (ig_time_mean_low < -time_low_thres),
    ig_time_mean_low,
    0,
)

# filter high example
space_high_thres = np.percentile(np.abs(ig_space_mean[1, -1, :]), 99.9)
time_high_thres = np.percentile(np.abs(ig_time_mean[1, -1, :]), 99.9)

ig_space_mean_high_filt = np.where(
    (ig_space_mean_high > space_high_thres) | (ig_space_mean_high < -space_high_thres),
    ig_space_mean_high,
    0,
)

ig_time_mean_high_filt = np.where(
    (ig_time_mean_high > time_high_thres) | (ig_time_mean_high < -time_high_thres),
    ig_time_mean_high,
    0,
)

ig_space_filt = np.stack([ig_space_mean_low_filt, ig_space_mean_high_filt])
ig_time_filt = np.stack([ig_time_mean_low_filt, ig_time_mean_high_filt])


# sum up ig scores over energy and water variables for all four cases
ig_space_energy = (
    ig_space_filt[:, 0, :] + ig_space_filt[:, 6, :] + ig_space_filt[:, 7, :]
)
ig_space_energy = np.abs(ig_space_energy)
ig_space_water = (
    ig_space_filt[:, 1, :]
    + ig_space_filt[:, 2, :]
    + ig_space_filt[:, 4, :]
    + ig_space_filt[:, 5, :]
)
ig_space_water = np.abs(ig_space_water)

ig_time_energy = ig_time_filt[:, 0, :] + ig_time_filt[:, 6, :] + ig_time_filt[:, 7, :]
ig_time_water = (
    ig_time_filt[:, 1, :]
    + ig_time_filt[:, 2, :]
    + ig_time_filt[:, 4, :]
    + ig_time_filt[:, 5, :]
)

# calculate the diff between both models for the low example and cut of the first year
ig_diff_water_low = np.abs(ig_space_water[0,:] - ig_time_water[0,:])[:,365:]
ig_diff_energy_low = np.abs(ig_space_energy[0,:] - ig_time_energy[0,:])[:,365:]

# create the same colorbar for water and energy
vmax = np.max([ig_diff_energy_low.max(), ig_diff_water_low.max()])
print([ig_diff_energy_low.max(), ig_diff_water_low.max()])
levels = np.linspace(0, vmax, 5).round(decimals=7)

# mask values of zero with nan
ig_diff_energy_low = np.where(ig_diff_energy_low != 0, ig_diff_energy_low, np.nan)

# set lowest value of colormap to zero

# Get the colormap and convert it to a list
cmap_reds = plt.cm.Reds
new_cmap_reds = cmap_reds(np.linspace(0, 1, len(levels)))  # Extract colors

cmap_blues = plt.cm.Blues
new_cmap_blues = cmap_blues(np.linspace(0, 1, len(levels)))  # Extract colors

# Make the first (lowest) color fully transparent
new_cmap_reds[0, -1] = 0  # Set alpha to 0
new_cmap_blues[0, -1] = 0  # Set alpha to 0

# Create a new colormap from modified colors
transparent_cmap_reds = mcolors.ListedColormap(new_cmap_reds)
transparent_cmap_blues = mcolors.ListedColormap(new_cmap_blues)

# plot low example
plot = ax2.contourf(
    dates,
    periods,
    ig_diff_energy_low,
    vmax=vmax,
    vmin=0,
    cmap=transparent_cmap_reds,
    extend="max",
    levels=levels,
)
cbar1 = fig.colorbar(plot, ax=ax2, extendrect=False, location="bottom", orientation="horizontal", label="Energy Importance", pad=0.05)

for c in plot.collections:
    c.set_edgecolor("face")

# mask values of zero with nan
ig_diff_water_low = np.where(ig_diff_water_low != 0, ig_diff_water_low, np.nan)

plot = ax2.contourf(
    dates,
    periods,
    ig_diff_water_low,
    cmap=transparent_cmap_blues,
    vmax=vmax,
    vmin=0,
    levels=levels,
    extend="max",
)
cbar2 = fig.colorbar(plot, ax=ax2, extendrect=False, location="bottom", orientation="horizontal", label="Water Importance", pad=0.03)
# Manually adjust position of the second colorbar
# pos1 = cbar1.ax.get_position()
# pos2 = cbar2.ax.get_position()
# cbar2.ax.set_position([pos2.x0, 0, pos2.width, pos2.height])  

for c in plot.collections:
    c.set_edgecolor("face")

#ax2.set_xlabel("Time (months)")
ax2.set_ylabel("Period (days)")
ax2.set_yscale("log")

ax2.vlines(
    start_index_low, 0, 730, colors="black", linestyles="--", alpha=alpha_vlines
)
ax2.vlines(
    end_index_low, 0, 730, colors="black", linestyles="--", alpha=alpha_vlines
)

ax2.set_xticks(month_starts)
ax2.set_xticklabels(month_names)

## high example plots
ig_diff_water_high = np.abs(ig_space_water[1,:] - ig_time_water[1,:])[:,365:]
ig_diff_energy_high = np.abs(ig_space_energy[1,:] - ig_time_energy[1,:])[:,365:]

vmax = np.max([ig_diff_energy_high.max(), ig_diff_water_high.max()])
levels = np.linspace(0, vmax, 5).round(decimals=7)

# mask values of zero with nan
ig_diff_energy_high = np.where(ig_diff_energy_high != 0, ig_diff_energy_high, np.nan)
ig_diff_water_high = np.where(ig_diff_water_high != 0, ig_diff_water_high, np.nan)

plot = ax3.contourf(
    dates,
    periods,
    ig_diff_energy_high,
    cmap=transparent_cmap_reds,
    vmax=vmax,
    vmin=0,
    levels=levels,
    extend="max",
)
cbar = fig.colorbar(plot, ax=ax3, location="bottom", orientation="horizontal", label="Energy Importance", pad=0.05)

for c in plot.collections:
    c.set_edgecolor("face")

plot = ax3.contourf(
    dates,
    periods,
    ig_diff_water_high,
    cmap=transparent_cmap_blues,
    vmax=vmax,
    vmin=0,
    extend="max",
    levels=levels,
)
cbar = fig.colorbar(plot, ax=ax3, location="bottom", orientation="horizontal", label="Water Importance", pad=0.03)

for c in plot.collections:
    c.set_edgecolor("face")

#ax3.set_xlabel("Time (months)")
ax3.set_yscale("log")

ax3.vlines(
    start_index_high, 0, 730, colors="black", linestyles="--", alpha=alpha_vlines
)
ax3.vlines(
    end_index_high, 0, 730, colors="black", linestyles="--", alpha=alpha_vlines
)

ax3.set_xticks(month_starts)
ax3.set_xticklabels(month_names)


#fig.tight_layout()
fig.savefig(f"{save_path}/case_study_figure3_ver2.png", dpi=300)
# %%
# space - time hinzufügen