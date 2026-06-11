"""
Plot appendix figure S1: Test performance per test site (gridcell)

"""

# %%

import pandas as pd
import matplotlib.pyplot as plt

# %%
path = f"../trained_models/version1/results/model_performance"

model_set = "time"

time_r2iav = pd.read_csv(f"{path}/{model_set}_r2iav_site.csv", index_col=0, header=0)
time_r2iav_2 = pd.read_csv(
    f"{path}/{model_set}_r2iav_2_site.csv", index_col=0, header=0
)
time_r2anom = pd.read_csv(f"{path}/{model_set}_r2anom_site.csv", index_col=0, header=0)

time_r2_general = pd.read_csv(f"{path}/{model_set}_perf_r2_site_year.csv")

model_set = "space"

space_r2iav = pd.read_csv(f"{path}/{model_set}_r2iav_site.csv", index_col=0, header=0)
space_r2iav_2 = pd.read_csv(
    f"{path}/{model_set}_r2iav_2_site.csv", index_col=0, header=0
)
space_r2anom = pd.read_csv(f"{path}/{model_set}_r2anom_site.csv", index_col=0, header=0)

space_r2_general = pd.read_csv(f"{path}/{model_set}_perf_r2_site_year.csv")


model_set = "timespace"

timespace_r2iav = pd.read_csv(
    f"{path}/{model_set}_r2iav_site.csv", index_col=0, header=0
)
timespace_r2iav_2 = pd.read_csv(
    f"{path}/{model_set}_r2iav_2_site.csv", index_col=0, header=0
)
timespace_r2anom = pd.read_csv(
    f"{path}/{model_set}_r2anom_site.csv", index_col=0, header=0
)

timespace_r2_general = pd.read_csv(f"{path}/{model_set}_perf_r2_site_year.csv")


# %%

# calculate average across different models
space_r2_iav_mean = space_r2iav.mean(axis=1)
space_r2_site_mean = space_r2_general.loc[:, "site_mean"]

time_r2_iav_mean = time_r2iav.mean(axis=1)
time_r2_site_mean = time_r2_general.loc[:, "site_mean"]

space_r2_iav_mean = space_r2iav.mean(axis=1)
space_r2_site_mean = space_r2_general.loc[:, "site_mean"]

timespace_r2_iav_mean = timespace_r2iav.mean(axis=1)
timespace_r2_site_mean = timespace_r2_general.loc[:, "site_mean"]


# %%

# Set plot style
plt.rcParams["font.family"] = "TeX Gyre PagellaX"
plt.rcParams["font.size"] = 11
plt.rcParams["xtick.major.size"] = 5  # X-axis major ticks
plt.rcParams["ytick.major.size"] = 5

width_in = 130 / 25.4
height_in = 110 / 25.4

fig = plt.figure(figsize=(width_in, height_in))
ax = fig.add_subplot()

ax.scatter(space_r2_site_mean, space_r2_iav_mean, label="Space", color="#c6d325")
ax.scatter(time_r2_site_mean, time_r2_iav_mean, label="Time", color="#ef7c00")
ax.scatter(
    timespace_r2_site_mean, timespace_r2_iav_mean, label="Time+space", color="#00b1ea"
)

ax.legend(loc="center left", frameon=True)
ax.set_xlim(0.4, 0.95)
# ax.set_xlim(-1.2, 1)
ax.set_ylim(-1.1, 1)
ax.set_xlabel("$R^2$")
ax.set_ylabel("$R^2_{IAV}$")

fig.tight_layout()
fig.savefig(f"{path}/test_site_r2_r2iav_model_comp.png", dpi=600)
# %%

import seaborn as sns
from matplotlib.gridspec import GridSpec

plt.rcParams["font.size"] = 10

# Create figure with custom grid layout
fig = plt.figure(figsize=(width_in, height_in))
gs = GridSpec(
    3, 3, height_ratios=[1, 4, 0.2], width_ratios=[0.2, 4, 1], hspace=0.11, wspace=0.11
)

# Main scatter plot (center)
ax_main = fig.add_subplot(gs[1, 1])
ax_main.scatter(
    space_r2_site_mean, space_r2_iav_mean, label="Space", color="#c6d325", alpha=0.7
)
ax_main.scatter(
    time_r2_site_mean, time_r2_iav_mean, label="Time", color="#ef7c00", alpha=0.7
)
ax_main.scatter(
    timespace_r2_site_mean,
    timespace_r2_iav_mean,
    label="Time+space",
    color="#00b1ea",
    alpha=0.7,
)
ax_main.set_xlim(0.4, 0.95)
ax_main.set_ylim(-1.1, 1)
ax_main.set_xlabel("$R^2$")
ax_main.set_ylabel("$R^2_{IAV}$")
ax_main.legend(loc="center left", frameon=True, handletextpad=0.2, labelspacing=0.3)

# Top KDE (x-axis)
ax_top = fig.add_subplot(gs[0, 1], sharex=ax_main)
sns.kdeplot(x=space_r2_site_mean, ax=ax_top, color="#c6d325", alpha=0.6, label="Space")
sns.kdeplot(x=time_r2_site_mean, ax=ax_top, color="#ef7c00", alpha=0.6, label="Time")
sns.kdeplot(
    x=timespace_r2_site_mean, ax=ax_top, color="#00b1ea", alpha=0.6, label="Time+space"
)
ax_top.tick_params(labelbottom=False)
ax_top.set_xlabel("")
ax_top.set_ylabel("Density")

# Right KDE (y-axis)
ax_right = fig.add_subplot(gs[1, 2], sharey=ax_main)
sns.kdeplot(y=space_r2_iav_mean, ax=ax_right, color="#c6d325", alpha=0.6)
sns.kdeplot(y=time_r2_iav_mean, ax=ax_right, color="#ef7c00", alpha=0.6)
sns.kdeplot(y=timespace_r2_iav_mean, ax=ax_right, color="#00b1ea", alpha=0.6)
ax_right.tick_params(labelleft=False)
ax_right.set_xlabel("Density")

# fig.tight_layout()
fig.savefig(f"{path}/jointplot_r2_r2iav_model_comp.png", dpi=600)


# %%
