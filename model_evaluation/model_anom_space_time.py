"""
Calculate IG values to run script: figure4_integrated_gradients_anom.py

1. Extract anomaly years per pixel from the target data
2. Create scatter plot with R^2 between the target nep sum and the models prediction
3. Calculate IG for those anomaly years for both models
"""


# %%
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score
from scipy.stats import linregress
from ml_models import Model18_IG_sum
from captum.attr import IntegratedGradients

#%%
# load data
load_path_nepsum = f"../trained_models/version1/results"

space_nepsum = np.load(f"{load_path_nepsum}/nep_sum_space_siteyear.npy")
time_nepsum = np.load(f"{load_path_nepsum}/nep_sum_time_siteyear.npy")
target_nepsum = np.load(f"{load_path_nepsum}/nep_sum_target_siteyear.npy")

# load the total predictions
space_nep = np.load(f"{load_path_nepsum}/nep_model_space_siteyear.npy")
time_nep = np.load(f"{load_path_nepsum}/nep_model_time_siteyear.npy")
target_nep = np.load(f"{load_path_nepsum}/nep_target_siteyear.npy")

# load values for denormalisation
load_path_denorm = "../data/pre_processing_steps"
tar_max_denrom = np.load(f"{load_path_denorm}/tar_max_normval.npy")
tar_min_denorm = np.load(f"{load_path_denorm}/tar_min_normval.npy")

save_path = f"../trained_models/version1/ig_results_anom"

#%%
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
    return y_norm * (y_max - y_min) + y_min

space_nep_denorm = denormalize_minmax(space_nep, tar_min_denorm[0], tar_max_denrom[0])
time_nep_denorm = denormalize_minmax(time_nep, tar_min_denorm[0], tar_max_denrom[0])
target_nep_denorm = denormalize_minmax(target_nep, tar_min_denorm[0], tar_max_denrom[0])

# calculate the sum
space_denorm_sum = space_nep_denorm[:,:,0,:].sum(axis=(2))
time_denorm_sum = time_nep_denorm[:,:,0,:].sum(axis=(2))
target_denorm_sum = target_nep_denorm[:,:,0,:].sum(axis=(2))

#%%
# Initialize lists to collect data points for all sites
target_low = []
space_low = []
time_low = []

target_low_denorm = []
space_low_denorm = []
time_low_denorm = []

space_index = []
time_index = []

perc = 99

# Loop over each site (axis=0; 30 sites)
for site in range(target_nepsum.shape[0]):
    # Compute the 10th percentile for the target NEP at this site
    perc10 = np.percentile(target_nepsum[site, :], perc)
    
    # Get the indices (years) where the target is below the 10th percentile
    indices = np.where(target_nepsum[site, :] > perc10)[0]
    
    # Extract the data for these indices and add to our lists
    target_low.extend(target_nepsum[site, indices])
    space_low.extend(space_nepsum[site, indices])
    time_low.extend(time_nepsum[site, indices])

    target_low_denorm.extend(target_denorm_sum[site, indices])
    space_low_denorm.extend(space_denorm_sum[site, indices])
    time_low_denorm.extend(time_denorm_sum[site, indices])

    space_index.append(indices)
    time_index.append(indices)

space_anom_index = np.stack(space_index)
time_anom_index = np.stack(time_index)

# Convert lists to NumPy arrays for further processing
target_low = np.array(target_low)
space_low = np.array(space_low)
time_low = np.array(time_low)

target_low_denorm = np.array(target_low_denorm)
space_low_denorm = np.array(space_low_denorm)
time_low_denorm = np.array(time_low_denorm)

num_sites, num_years = space_anom_index.shape  # (30,3)

np.save(f"{save_path}/anom_{perc}_nep_target_{num_years}.npy", target_low)
np.save(f"{save_path}/anom_{perc}_nep_space_{num_years}.npy", space_low)
np.save(f"{save_path}/anom_{perc}_nep_time_{num_years}.npy", time_low)

np.save(f"{save_path}/anom_{perc}_denormnep_target_{num_years}.npy", target_low_denorm)
np.save(f"{save_path}/anom_{perc}_denormnep_space_{num_years}.npy", space_low_denorm)
np.save(f"{save_path}/anom_{perc}_denormnep_time_{num_years}.npy", time_low_denorm)

# Calculate R^2 values comparing each model with the target
r2_space = r2_score(target_low, space_low)
r2_time = r2_score(target_low, time_low)

# Also perform linear regressions so we can plot regression lines
slope_space, intercept_space, _, _, _ = linregress(space_low, target_low)
slope_time, intercept_time, _, _, _ = linregress(time_low, target_low)

# Set plot style
plt.rcParams["font.family"] = "TeX Gyre PagellaX"
plt.rcParams["font.size"] = 11
plt.rcParams['xtick.major.size'] = 5  # X-axis major ticks
plt.rcParams['ytick.major.size'] = 5

# Create a scatter plot of the modeled vs. target NEP values for the low-percentile years
plt.figure(figsize=(4, 4))

# Scatter plot for the space model
plt.scatter(space_low, target_low, color="#c6d325", alpha=0.8, edgecolors="black", 
            label=f'Space Model (R² = {r2_space:.2f})')
# Scatter plot for the time model
plt.scatter(time_low, target_low, color="#ef7c00", alpha=0.8, edgecolors="black",
            label=f'Time Model (R² = {r2_time:.2f})')

# Prepare x-values for plotting the regression lines.
# We use the overall min and max from the two models so that both lines are plotted over a common range.
x_min = min(space_low.min(), time_low.min())
x_max = max(space_low.max(), time_low.max())
x_vals = np.linspace(x_min, x_max, 100)

# Plot the regression line for the space model
plt.plot(x_vals, slope_space * x_vals + intercept_space, color="#c6d325",
         linestyle='--', label='Space Regression')
# Plot the regression line for the time model
plt.plot(x_vals, slope_time * x_vals + intercept_time, color="#ef7c00",
         linestyle='--', label='Time Regression')

# Plot the 1:1 line (identity line)
plt.plot([225, 250], [225, 250], color='black', linestyle='--', linewidth=1.5)

plt.xlim(225,250)
plt.ylim(225,250)
# Labeling the plot
plt.xlabel('Modeled NEP Sum')
plt.ylabel('Target NEP Sum (Normalized)')
#plt.title('Scatter Plot of NEP Sum (Years Below 1th Percentile per Site)')
plt.legend(frameon=False)
plt.tight_layout()
plt.savefig(f"{save_path}/scatter_anom_{perc}_mod_target_nep.png", dpi=300)
plt.show()

# %%

# create dataframe for IG calculation

num_sites, num_years = space_anom_index.shape  # (30,3)

# Create the site index: repeat each site number 3 times
site_indices = np.repeat(np.arange(num_sites), num_years)

# Flatten the space_anom_index array to get the corresponding year indices
year_indices = space_anom_index.flatten()

# Create the DataFrame
df_anom_years = pd.DataFrame({'site': site_indices, 'year': year_indices})

# add year sum to dataframe
df_anom_years["Target Sum"] = target_low
df_anom_years["Space Sum"] = space_low
df_anom_years["Time Sum"] = time_low


#%%
# IG part 

# load test data
device = torch.device("cuda")
test_data_path = "../data/datasets/test_sets"
# total test
test_feat = torch.from_numpy(np.load(f"{test_data_path}/test2_test_feat.npy")).float()
test_stat = torch.from_numpy(np.load(f"{test_data_path}/test2_test_stat.npy")).float()
# extract the last 30 years
test_feat = test_feat[:,-30:,:].to(device)
test_stat = test_stat[:,-30:,:].to(device)

# calculate IG for the specific anomaly years for both models
num_years=30
# load the models

n_epochs = 300
# init model
device = torch.device("cuda")
num_input = 11  
num_output = 365 * 2
model_space = Model18_IG_sum(num_input, num_output).to(device)
model_time = Model18_IG_sum(num_input, num_output).to(device)


model_set = "space"
model_path = f"../trained_models/version1"


def calc_ig(ig, sitey_feat, sitey_stat, site_feat_mean, site_stat):
    """
    Calculate IG
    """
    attribution = ig.attribute(inputs=(sitey_feat, sitey_stat), target=0, baselines=(site_feat_mean, site_stat))

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
ig_space = np.zeros((5,num_years,11,64,730))
ig_time = np.zeros((5,num_years,11,64,730))

# loop over the model initalisations
for i in range(5):
    print(i)
    # load space model
    model_set = "space"
    model_code = f"resnet18_{model_set}_1"
    modelinit_path = f"{model_path}/space_set/6_{model_code}_{n_epochs}/init_{i}/m{model_code}.pt"
    model_space.load_state_dict(torch.load(modelinit_path, map_location=torch.device('cpu')))

    # load time model
    model_set = "time"
    model_code = f"resnet18_{model_set}_1"
    modelinit_path = f"{model_path}/time_set/6_{model_code}_{n_epochs}/init_{i}/m{model_code}.pt"
    model_time.load_state_dict(torch.load(modelinit_path, map_location=torch.device('cpu')))
    
    igmod_space = IntegratedGradients(model_space)
    igmod_time = IntegratedGradients(model_time)

    # loop through the data
    for j in range(num_years):
        
        index_site = int(df_anom_years.loc[j,"site"])
        index_year = int(df_anom_years.loc[j,"year"])
        # index_site = int(bottom_entries_df.loc[j,"site"])
        # index_year = int(bottom_entries_df.loc[j,"year"])
        
        sample_feat = test_feat[index_site, index_year, :]
        sample_feat = sample_feat[None, :]
        sample_stat = test_stat[index_site, index_year, :]
        sample_stat = sample_stat[None, :]
        
        # calculate baseline for IG
        site_mean_feat = test_feat[index_site, :].mean(axis=0)
        site_mean_feat = site_mean_feat[None, :]
        site_mean_stat = test_stat[index_site, :].mean(axis=0)
        site_mean_stat = site_mean_stat[None, :]
        # calculate IG
        ig_space[i,j,:] = calc_ig(igmod_space, sample_feat, sample_stat, site_mean_feat, site_mean_stat)
        ig_time[i,j,:] = calc_ig(igmod_time, sample_feat, sample_stat, site_mean_feat, site_mean_stat)
        
# calculate mean over models
ig_space_mean = ig_space.mean(axis=0)
ig_time_mean = ig_time.mean(axis=0)

# save IG results to disk
np.save(f"{save_path}/anom_{perc}_ig_space_mean_{num_years}.npy", ig_space_mean)
np.save(f"{save_path}/anom_{perc}_ig_time_mean_{num_years}.npy", ig_time_mean)
df_anom_years.to_csv(f"{save_path}/anom_{perc}_years_{num_years}.csv")

# %%
