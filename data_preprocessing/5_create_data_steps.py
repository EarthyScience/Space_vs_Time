"""
Preprocessing Step 5:

Load pre-selected numpy files and create six processing steps for each (train/validation/test) split.

"""

# %%
import numpy as np
from pathlib import Path
import os

# %%
np.random.seed(42)

# Root directory for all dataset outputs (set via environment variable or default local path)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

save_path_basic = DATA_ROOT / "datasets" / "basic_obs"
save_path_space_set = DATA_ROOT / "datasets" / "space_set"
save_path_time_set = DATA_ROOT / "datasets" / "time_set"
save_path_timespace_set = DATA_ROOT / "datasets" / "timespace_set"

# %%
### 1. Time set steps

# load data
train_total_feat = np.load(save_path_time_set / "basic_train_total_feat.npy")
train_total_stat = np.load(save_path_time_set / "basic_train_total_stat.npy")
train_total_tar  = np.load(save_path_time_set / "basic_train_total_tar.npy")

val_total_feat = np.load(save_path_time_set / "basic_val_total_feat.npy")
val_total_stat = np.load(save_path_time_set / "basic_val_total_stat.npy")
val_total_tar  = np.load(save_path_time_set / "basic_val_total_tar.npy")

# load basic data
basic_train_feat = np.load(save_path_basic / "basic_train_feat.npy")
basic_train_stat = np.load(save_path_basic / "basic_train_stat.npy")
basic_train_tar  = np.load(save_path_basic / "basic_train_tar.npy")

basic_val_feat = np.load(save_path_basic / "basic_val_feat.npy")
basic_val_stat = np.load(save_path_basic / "basic_val_stat.npy")
basic_val_tar  = np.load(save_path_basic / "basic_val_tar.npy")
#%%
# create steps
num_years = [20, 40, 60, 80, 100, 120]
for i in range(6):
    years = num_years[i]

    train_feat = train_total_feat[:, -years:, :].reshape([10*years, 11, 64, 730])
    train_stat = train_total_stat[:, -years:, :].reshape([10*years, 3])
    train_tar = train_total_tar[:, -years:, :].reshape([10*years, 2, 365])

    val_feat = val_total_feat[:, -years:, :].reshape([2*years, 11, 64, 730])
    val_stat = val_total_stat[:, -years:, :].reshape([2*years, 3])
    val_tar = val_total_tar[:, -years:, :].reshape([2*years, 2, 365])

    np.save(f"{save_path_time_set}/time_{i+1}_train_feat.npy", train_feat)
    np.save(f"{save_path_time_set}/time_{i+1}_train_stat.npy", train_stat)
    np.save(f"{save_path_time_set}/time_{i+1}_train_tar.npy", train_tar)

    np.save(f"{save_path_time_set}/time_{i+1}_val_feat.npy", val_feat)
    np.save(f"{save_path_time_set}/time_{i+1}_val_stat.npy", val_stat)
    np.save(f"{save_path_time_set}/time_{i+1}_val_tar.npy", val_tar)

#%%

### 2. Space set steps
# load data
train_feat = np.load(f"{save_path_space_set}/space_train_feat.npy")
train_stat = np.load(f"{save_path_space_set}/space_train_stat.npy")
train_tar = np.load(f"{save_path_space_set}/space_train_tar.npy")

val_feat = np.load(f"{save_path_space_set}/space_val_feat.npy")
val_stat = np.load(f"{save_path_space_set}/space_val_stat.npy")
val_tar = np.load(f"{save_path_space_set}/space_val_tar.npy")
#%%
# Number of sites
num_sites = 150
step_size = 25
step_sites_num = [15, 40, 65, 90, 115, 140]
# Random permutation of sites
random_indices = np.random.permutation(num_sites)

# Create 6 sub-data splits
for step in range(1, 7):
    num_sites_in_split = step_sites_num[step-1]
    selected_indices = random_indices[:num_sites_in_split]
    print(selected_indices)
    # Extract the corresponding data
    train_feat_split = train_feat[selected_indices].reshape([num_sites_in_split*8, 11, 64, 730])
    train_stat_split = train_stat[selected_indices].reshape([num_sites_in_split*8, 3])
    train_tar_split = train_tar[selected_indices].reshape([num_sites_in_split*8, 365, 2])

    train_feat_split = np.concatenate([basic_train_feat, train_feat_split])
    train_stat_split = np.concatenate([basic_train_stat, train_stat_split])
    train_tar_split = np.concatenate([basic_train_tar, train_tar_split])
    # Save each split
    np.save(f"{save_path_space_set}/space_{step}_train_feat.npy", train_feat_split)
    np.save(f"{save_path_space_set}/space_{step}_train_stat.npy", train_stat_split)
    np.save(f"{save_path_space_set}/space_{step}_train_tar.npy", train_tar_split)

#%%
# Number of sites
num_sites = val_feat.shape[0]
step_size = 3
step_sites_num = [1, 4, 7, 10, 13, 16]
# Random permutation of sites
random_indices = np.random.permutation(num_sites)

# Create 6 sub-data splits for validation
for step in range(1, 7):
    num_sites_in_split = step_sites_num[step-1]
    selected_indices = random_indices[:num_sites_in_split]
    print(selected_indices)
    # Extract the corresponding data
    val_feat_split = val_feat[selected_indices].reshape([num_sites_in_split*8, 11, 64, 730])
    val_stat_split = val_stat[selected_indices].reshape([num_sites_in_split*8, 3])
    val_tar_split = val_tar[selected_indices].reshape([num_sites_in_split*8, 365, 2])

    val_feat_split = np.concatenate([basic_val_feat, val_feat_split])
    val_stat_split = np.concatenate([basic_val_stat, val_stat_split])
    val_tar_split = np.concatenate([basic_val_tar, val_tar_split])

    np.save(f"{save_path_space_set}/space_{step}_val_feat_sites.npy", val_feat_split)
    np.save(f"{save_path_space_set}/space_{step}_val_stat_sites.npy", val_stat_split)
    np.save(f"{save_path_space_set}/space_{step}_val_tar_sites.npy", val_tar_split)

# %%

## space and time steps
# load data
train_feat = np.load(f"{save_path_timespace_set}/timespace_train_feat.npy")
train_stat = np.load(f"{save_path_timespace_set}/timespace_train_stat.npy")
train_tar = np.load(f"{save_path_timespace_set}/timespace_train_tar.npy")

val_feat = np.load(f"{save_path_timespace_set}/timespace_val_feat.npy")
val_stat = np.load(f"{save_path_timespace_set}/timespace_val_stat.npy")
val_tar = np.load(f"{save_path_timespace_set}/timespace_val_tar.npy")


train_feat = np.concatenate([train_total_feat[:,-35:,:], train_feat])
train_stat = np.concatenate([train_total_stat[:,-35:,:], train_stat])
train_tar = np.concatenate([train_total_tar[:,-35:,:], train_tar])

# slice basic obs validation sites to validation
val_feat = np.concatenate([val_total_feat[:,-20:,:], val_feat])
val_stat = np.concatenate([val_total_stat[:,-20:,:], val_stat])
val_tar = np.concatenate([val_total_tar[:,-20:,:], val_tar])

#%%
# Number of sites
num_sites = 34
step_sites_num = [14, 20, 24, 28, 32, 34]
step_year_num = [14, 20, 25, 29, 32, 35]

# Random permutation of sites
random_indices = np.random.permutation(num_sites)
# the first ten sites should always be the basic training sites
random_indices = np.concatenate([np.arange(0,10), random_indices + 10])

# Create 6 sub-data splits
for step in range(1, 7):
    num_sites_in_split = step_sites_num[step-1]
    num_years_in_split = step_year_num[step-1]
    selected_indices = random_indices[:num_sites_in_split]

    print(selected_indices)

    # Extract the corresponding data
    train_feat_split = train_feat[selected_indices]
    train_stat_split = train_stat[selected_indices]
    train_tar_split = train_tar[selected_indices]
    
    if step == 1:
        # select 14 random years
        selected_years = np.random.choice(20, size=14, replace=False)
        train_feat_split = train_feat_split[:,selected_years].reshape([num_sites_in_split*14, 11, 64, 730])
        train_stat_split = train_stat_split[:,selected_years].reshape([num_sites_in_split*14, 3])
        train_tar_split = train_tar_split[:,selected_years].reshape([num_sites_in_split*14, 365, 2])
    else:
        train_feat_split = train_feat_split[:,-num_years_in_split:].reshape([num_sites_in_split*num_years_in_split, 11, 64, 730])
        train_stat_split = train_stat_split[:,-num_years_in_split:].reshape([num_sites_in_split*num_years_in_split, 3])
        train_tar_split = train_tar_split[:,-num_years_in_split:].reshape([num_sites_in_split*num_years_in_split, 365, 2])

    print(train_feat_split.shape)
    # Save each split
    np.save(f"{save_path_timespace_set}/timespace_{step}_train_feat.npy", train_feat_split)
    np.save(f"{save_path_timespace_set}/timespace_{step}_train_stat.npy", train_stat_split)
    np.save(f"{save_path_timespace_set}/timespace_{step}_train_tar.npy", train_tar_split)

## for validation
#%%
# Number of sites
num_sites = 7 # seven additional sites
step_sites_num = [3, 5, 6, 7, 8, 9]
step_year_num = [7, 8, 10, 12, 13, 14]

# Random permutation of sites
random_indices = np.random.permutation(num_sites)
# the first two sites should always be the basic validation sites
random_indices = np.concatenate([np.array([0,1]), random_indices + 2])
random_years_indices = np.random.permutation(14)
# Create 6 sub-data splits
for step in range(1, 7):
    num_sites_in_split = step_sites_num[step-1]
    num_years_in_split = step_year_num[step-1]
    selected_indices = random_indices[:num_sites_in_split]

    print(step, selected_indices)

    # Extract the corresponding data
    val_feat_split = val_feat[selected_indices]
    val_stat_split = val_stat[selected_indices]
    val_tar_split = val_tar[selected_indices]
    
    selected_years = random_years_indices[:num_years_in_split]
    val_feat_split = val_feat_split[:, selected_years].reshape([num_sites_in_split*num_years_in_split, 11, 64, 730])
    val_stat_split = val_stat_split[:, selected_years].reshape([num_sites_in_split*num_years_in_split, 3])
    val_tar_split = val_tar_split[:, selected_years].reshape([num_sites_in_split*num_years_in_split, 365, 2])

    print(val_feat_split.shape)
    # Save each split
    np.save(f"{save_path_timespace_set}/timespace_{step}_val_feat.npy", val_feat_split)
    np.save(f"{save_path_timespace_set}/timespace_{step}_val_stat.npy", val_stat_split)
    np.save(f"{save_path_timespace_set}/timespace_{step}_val_tar.npy", val_tar_split)


# %%
