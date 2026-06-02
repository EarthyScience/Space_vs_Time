"""
Produce figure 4:

"""

#%%

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import linregress
#%%
path = f"../trained_models/version1/ig_results_anom"

save_path = "../trained_models/version1/ig_results/plots"

perc = [1, 99]
num_years = 30

anom_ig_space_mean = {p: np.load(f"{path}/anom_{p}_ig_space_mean_{num_years}.npy") for p in perc}
anom_ig_time_mean = {p: np.load(f"{path}/anom_{p}_ig_time_mean_{num_years}.npy") for p in perc}

anom_nep_target = {p: np.load(f"{path}/anom_{p}_denormnep_target_1.npy") for p in perc}
anom_nep_space = {p: np.load(f"{path}/anom_{p}_denormnep_space_1.npy") for p in perc}
anom_nep_time = {p: np.load(f"{path}/anom_{p}_denormnep_time_1.npy") for p in perc}

#%%

def convert_flux_kgCO2_to_gC_per_m2_yr(flux_kgCO2_per_m2_s):
    """
    Converts flux from kg(CO2) / (m^2 * s) to gC / (m^2 * yr).

    Parameters:
    - flux_kgCO2_per_m2_s: NumPy array or scalar, flux in kg CO2 per (m^2 * s)

    Returns:
    - Flux in gC per (m^2 * yr)
    """
    conversion_factor = 2.355*1e7  # From kg(CO2)/(m^2*s) to gC/(m^2*yr)
    return flux_kgCO2_per_m2_s * conversion_factor

target_low = convert_flux_kgCO2_to_gC_per_m2_yr(anom_nep_target[1])
target_high = convert_flux_kgCO2_to_gC_per_m2_yr(anom_nep_target[99])

space_low = convert_flux_kgCO2_to_gC_per_m2_yr(anom_nep_space[1])
space_high = convert_flux_kgCO2_to_gC_per_m2_yr(anom_nep_space[99])

time_low = convert_flux_kgCO2_to_gC_per_m2_yr(anom_nep_time[1])
time_high = convert_flux_kgCO2_to_gC_per_m2_yr(anom_nep_time[99])

r2_space_low = r2_score(target_low, space_low)
r2_space_high = r2_score(target_high, space_high)
r2_time_low = r2_score(target_low, time_low)
r2_time_high = r2_score(target_high, time_high)

# Also perform linear regressions so we can plot regression lines
slope_space, intercept_space, _, _, _ = linregress(space_low, target_low)
slope_time, intercept_time, _, _, _ = linregress(time_low, target_low)

# Also perform linear regressions so we can plot regression lines
slope_space_high, intercept_space_high, _, _, _ = linregress(space_high, target_high)
slope_time_high, intercept_time_high, _, _, _ = linregress(time_high, target_high)

## calculate IG

# boxplot together
def calc_ig_sum_year(ig_data):
    # calculate the absolute mean 
    ig_anom_abs_mean = np.abs(ig_data).mean(axis=(0))
    # calculate upper threshold
    upper_thres = np.percentile(ig_anom_abs_mean[-1, :, :], 99.9)
    loweser_thres = - upper_thres
    print(upper_thres)

    ig_anom_sum_year = np.sum(ig_data, where=np.logical_or(ig_data < loweser_thres, ig_data > upper_thres), axis=(2,3))
    print(ig_anom_sum_year.shape)
    
    return ig_anom_sum_year

num_years = 30
type ="anom_1"

ig_space = np.load(f"{path}/{type}_ig_space_mean_{num_years}.npy")
ig_time = np.load(f"{path}/{type}_ig_time_mean_{num_years}.npy")

space_sum_year_low = calc_ig_sum_year(ig_space)
time_sum_year_low = calc_ig_sum_year(ig_time)

num_years = 30
type ="anom_99"

ig_space = np.load(f"{path}/{type}_ig_space_mean_{num_years}.npy")
ig_time = np.load(f"{path}/{type}_ig_time_mean_{num_years}.npy")

space_sum_year_high = calc_ig_sum_year(ig_space)
time_sum_year_high = calc_ig_sum_year(ig_time)



# %%

plt.rcParams["font.family"] = "TeX Gyre PagellaX"
plt.rcParams["font.size"] = 11
plt.rcParams['xtick.major.size'] = 5  # X-axis major ticks
plt.rcParams['ytick.major.size'] = 5

width_in = 160 / 25.4
height_in = 150 / 25.4

fig = plt.figure(figsize=(width_in, height_in))
ax = fig.subplots(2,2)

# Scatter plot for the space model
ax[0,0].scatter(space_low, target_low, color="#c6d325", alpha=0.8, edgecolors="black", 
            label=f'Space Model (R² = {r2_space_low:.2f})')
# Scatter plot for the time model
ax[0,0].scatter(time_low, target_low, color="#ef7c00", alpha=0.8, edgecolors="black",
            label=f'Time Model (R² = {r2_time_low:.2f})')

ax[0,1].scatter(space_high, target_high, color="#c6d325", alpha=0.8, edgecolors="black", 
            label=f'Space Model (R² = {r2_space_high:.2f})')
# Scatter plot for the time model
ax[0,1].scatter(time_high, target_high, color="#ef7c00", alpha=0.8, edgecolors="black",
            label=f'Time Model (R² = {r2_time_high:.2f})')


# Prepare x-values for plotting the regression lines.
# We use the overall min and max from the two models so that both lines are plotted over a common range.
x_min = min(space_low.min(), time_low.min())
x_max = max(space_low.max(), time_low.max())
x_vals = np.linspace(x_min, x_max, 100)
x_min = min(space_high.min(), time_high.min())
x_max = max(space_high.max(), time_high.max())
x_vals_high = np.linspace(x_min, x_max, 100)


# Plot the regression line for the space model
ax[0,0].plot(x_vals, slope_space * x_vals + intercept_space, color="#c6d325",
         linestyle='--')
# Plot the regression line for the time model
ax[0,0].plot(x_vals, slope_time * x_vals + intercept_time, color="#ef7c00",
         linestyle='--')

# Plot the regression line for the space model
ax[0,1].plot(x_vals_high, slope_space_high * x_vals_high + intercept_space_high, color="#c6d325",
         linestyle='--')
# Plot the regression line for the time model
ax[0,1].plot(x_vals_high, slope_time_high * x_vals_high + intercept_time_high, color="#ef7c00",
         linestyle='--')


# Plot the 1:1 line (identity line)
ax[0,0].plot([-142,32], [-142,32], color='black', linestyle='--', linewidth=1.5)
ax[0,1].plot([-1,132], [-1,132], color='black', linestyle='--', linewidth=1.5)

# set limitations
ax[0,0].set_xlim(-142,32)
ax[0,0].set_ylim(-142,32)

ax[0,1].set_xlim(-1,132)
ax[0,1].set_ylim(-1,132)
# Labeling the plot
ax[0,0].set_xlabel('Modeled NEP (gC m$^{-2}$ yr$^{-1}$)')
ax[0,0].set_ylabel('Target NEP (gC m$^{-2}$ yr$^{-1}$)' )
ax[0,1].set_xlabel('Modeled NEP (gC m$^{-2}$ yr$^{-1}$)')
#ax[0,0].set_ylabel('Target NEP Sum')
#plt.title('Scatter Plot of NEP Sum (Years Below 1th Percentile per Site)')
ax[0,0].legend(frameon=False, fontsize=9, handletextpad=0)
ax[0,1].legend(frameon=False, fontsize=9, handletextpad=0)

ax[0,0].set_title("Negative Anomalies")
ax[0,1].set_title("Positive Anomalies")
#### Plot IG plots

list_predictors_nc = ["ta", "precip", "hurs", "vpd", "sm", "snw", "shortwave", "longwave", "ba", "fapar", "rand"]

# create energy vs. water variable plot

# energy variables: temperature, vpd, shortwave, longwave
# water variables: precip, hurs, sm, snw,

ig_energy_space_low = space_sum_year_low[:, 0] + space_sum_year_low[:, 6] + space_sum_year_low[:, 7]
ig_energy_time_low = time_sum_year_low[:, 0] + time_sum_year_low[:, 6] + time_sum_year_low[:, 7]
ig_energy_diff_low = ig_energy_space_low - ig_energy_time_low

ig_water_space_low = space_sum_year_low[:, 1] + space_sum_year_low[:, 2] + space_sum_year_low[:, 4] + space_sum_year_low[:, 5]
ig_water_time_low = time_sum_year_low[:, 1] + time_sum_year_low[:, 2] + time_sum_year_low[:, 4] + time_sum_year_low[:, 5]
ig_water_diff_low = ig_water_space_low - ig_water_time_low

ig_energy_space_high = space_sum_year_high[:, 0] + space_sum_year_high[:, 6] + space_sum_year_high[:, 7]
ig_energy_time_high = time_sum_year_high[:, 0] + time_sum_year_high[:, 6] + time_sum_year_high[:, 7]
ig_energy_diff_high = ig_energy_space_high - ig_energy_time_high

ig_water_space_high = space_sum_year_high[:, 1] + space_sum_year_high[:, 2] + space_sum_year_high[:, 4] + space_sum_year_high[:, 5]
ig_water_time_high = time_sum_year_high[:, 1] + time_sum_year_high[:, 2] + time_sum_year_high[:, 4] + time_sum_year_high[:, 5]
ig_water_diff_high = ig_water_space_high - ig_water_time_high


import seaborn as sns

#cmap = sns.cubehelix_palette(start=1, light=1, as_cmap=True)

custom_cmap = sns.cubehelix_palette(
    start=1,     # shift hue to greenish-blue
    rot=0.5,      # reduce hue rotation to stay in green-teal range
    dark=0.1,      # how dark the darkest color is
    light=0.8,     # how light the lightest color is
    gamma=1.0,     # gamma correction
    as_cmap=True   # return a colormap, not list of colors
)

# For the "low" KDE
sns.kdeplot(
    x=ig_energy_diff_low, 
    y=ig_water_diff_low, 
    ax=ax[1,0], 
    fill=True,     # Fill contours
    cmap=custom_cmap,
    color="#006c66",                             # Choose a colormap
    levels=7,              # Number of contour levels
    #thresh=0.01,             # Minimum contour level to show
    linewidths=1,
    alpha=0.8
)
#ax[1,0].set_title("1% anom")
ax[1,0].scatter(ig_energy_diff_low, ig_water_diff_low, color="#006c66", alpha=0.8, edgecolors="black")

# For the "high" KDE
sns.kdeplot(
    x=ig_energy_diff_high, 
    y=ig_water_diff_high, 
    ax=ax[1,1], 
    fill=True,
    cmap=custom_cmap,
    color="#006c66",
    levels=7,
    linewidths=1,
    alpha=0.8
)
#ax[1,1].set_title("99% anom")
ax[1,1].scatter(ig_energy_diff_high, ig_water_diff_high, color="#006c66", alpha=0.8, edgecolors="black")

#ax[1,0].legend(loc=0)
#ax[1,1].legend(loc=0)
ax[1,0].set_xlabel("(Space - Time) Importance Energy")
ax[1,0].set_ylabel("(Space - Time) Importance Water")
ax[1,1].set_xlabel("(Space - Time) Importance Energy")

ax[1,0].set_xlim(-3,1.4)
ax[1,0].set_ylim(-3,1.4)

ax[1,1].set_xlim(-1.4,3.5)
ax[1,1].set_ylim(-1.4,3.5)


labels = ["A", "B", "C", "D"]
axes = [ax[0,0], ax[0,1], ax[1,0], ax[1,1]]

# Loop through each axis and add a large black letter
for ax, label in zip(axes, labels):
    ax.text(
        -0.1, 1.03,  # Position slightly outside the top-left corner
        label,
        transform=ax.transAxes,  # Use axis coordinates (0,0 bottom-left; 1,1 top-right)
        fontsize=16,  # Adjust size as needed
        fontweight="black",
        color="black"
    )


fig.tight_layout()
fig.savefig(f"{save_path}/scatter_anom_figure4.png", dpi=300)


# %%

# For the "high" KDE
sns.kdeplot(
    x=ig_energy_diff_high, 
    y=ig_water_diff_high, 
    fill=False,
    color="#006c66",
    levels=7,
    linewidths=1,
    alpha=0.8
)
#ax[1,1].set_title("99% anom")
plt.scatter(ig_energy_diff_high, ig_water_diff_high, color="#006c66", alpha=0.9)
plt.show()


#%%
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt

# For the "high" KDE
g = sns.jointplot(
    x=ig_energy_diff_high, 
    y=ig_water_diff_high, 
    fill=False,
    color="#006c66",
    levels=7,   
    #thresh=0.01,
    linewidths=1,
    kind="kde"
)

g.ax_joint.set_xlim(-1.4,3.5)
g.ax_joint.set_ylim(-1.4,3.5)

#%%
# as two in one row plot


plt.rcParams["font.family"] = "TeX Gyre PagellaX"
plt.rcParams["font.size"] = 11
plt.rcParams['xtick.major.size'] = 5  # X-axis major ticks
plt.rcParams['ytick.major.size'] = 5

width_in = 160 / 25.4
height_in = 80 / 25.4

fig = plt.figure(figsize=(width_in, height_in))
ax = fig.subplots(1,2)

# Scatter plot for the space model
ax[0].scatter(space_low, target_low, color="#c6d325", alpha=0.8, edgecolors="black", 
            label=f'Space Model (R² = {r2_space_low:.2f})')
# Scatter plot for the time model
ax[0].scatter(time_low, target_low, color="#ef7c00", alpha=0.8, edgecolors="black",
            label=f'Time Model (R² = {r2_time_low:.2f})')

ax[0].scatter(space_high, target_high, color="blue", alpha=0.8, edgecolors="black", 
            label=f'Space Model (R² = {r2_space_low:.2f})')
# Scatter plot for the time model
ax[0].scatter(time_high, target_high, color="green", alpha=0.8, edgecolors="black",
            label=f'Time Model (R² = {r2_time_low:.2f})')


# Prepare x-values for plotting the regression lines.
# We use the overall min and max from the two models so that both lines are plotted over a common range.
x_min = min(space_low.min(), time_low.min())
x_max = max(space_low.max(), time_low.max())
x_vals = np.linspace(x_min, x_max, 100)

# Plot the regression line for the space model
ax[0].plot(x_vals, slope_space * x_vals + intercept_space, color="#c6d325",
         linestyle='--', label='Space Regression')
# Plot the regression line for the time model
ax[0].plot(x_vals, slope_time * x_vals + intercept_time, color="#ef7c00",
         linestyle='--', label='Time Regression')

# Plot the 1:1 line (identity line)
ax[0].plot([-6.1e-6,6e-6], [-6.1e-6,6e-6], color='black', linestyle='--', linewidth=1.5)

ax[0].set_xlim(-6.1e-6,6e-6)
ax[0].set_ylim(-6.1e-6,6e-6)
# Labeling the plot
ax[0].set_xlabel('Modeled NEP Sum')
ax[0].set_ylabel('Target NEP Sum (Normalized)')
#plt.title('Scatter Plot of NEP Sum (Years Below 1th Percentile per Site)')
#ax[0,0].legend(frameon=False)
fig.tight_layout()
#fig.savefig(f"{save_path}/scatter_anom_{perc}_mod_target_nep.png", dpi=300)



# %%
