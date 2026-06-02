"""
Preprocessing Step 4:

Load dataframe for data splits and create numpy files with the corresponding data splits (train/validation/test).
"""

# %%
import os
from pathlib import Path
import numpy as np
import pandas as pd

# %%
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

# Load pixel split definitions
pixel_dir = DATA_ROOT / "pixel_information"

pixels_time_set = pd.read_csv(pixel_dir / "pixels_timeset.csv", index_col=0)
pixels_space_set = pd.read_csv(pixel_dir / "pixels_spaceset.csv", index_col=0)
pixels_spacetime_set = pd.read_csv(pixel_dir / "pixels_spacetimeset.csv", index_col=0)
obs_years = pd.read_csv(pixel_dir / "obs_sitelist_mainyears.csv")

# %%
# Root data directory (set by user or environment variable)
import os

DATA_ROOT = Path(
    os.environ.get("DATA_ROOT", Path("data"))
)

# Dataset output locations
save_path = DATA_ROOT / "datasets"
save_path_basic = save_path / "basic_obs"
save_path_space_set = save_path / "space_set"
save_path_time_set = save_path / "time_set"
save_path_timespace_set = save_path / "timespace_set"
save_path_test_sets = save_path / "test_sets"

# %%
# Preprocessed input data
load_path = DATA_ROOT / "pre_processing_steps"

feat = np.load(load_path / "allfeat_wt_norm.npy")
static = np.load(load_path / "all_stat_norm.npy")
targets = np.load(load_path / "all_tar_norm.npy")

print(feat.shape)
print(static.shape)
print(targets.shape)
# %%

# create basic split

# train basic split
pixel_basic = pixels_time_set[pixels_time_set['obs_set'] == True]

train_basic_feat = []
train_basic_stat = []
train_basic_tar = []

train_basic_total_feat = []
train_basic_total_stat = []
train_basic_total_tar = []

val_basic_feat= []
val_basic_stat = []
val_basic_tar = []

val_basic_total_feat= []
val_basic_total_stat = []
val_basic_total_tar = []

test_basic_feat = []
test_basic_stat = []
test_basic_tar = []

test_basic_total_feat= []
test_basic_total_stat = []
test_basic_total_tar = []

for row in pixel_basic.iterrows():
    index = row[0]
    dtype = row[1]["dset_type"]
    site_id = row[1]["obs_siteid"]
    print(dtype, site_id)

    if site_id == "extra_site":
        site_id = "IT-Ro2"

    # select all years
    sample_total_feat = feat[index, :, :]
    sample_total_stat = static[index, :, :]
    sample_total_tar = targets[index, :, :]

    if dtype == "train":
        train_basic_total_feat.append(sample_total_feat)
        train_basic_total_stat.append(sample_total_stat)
        train_basic_total_tar.append(sample_total_tar)
    elif dtype == "val":
        val_basic_total_feat.append(sample_total_feat)
        val_basic_total_stat.append(sample_total_stat)
        val_basic_total_tar.append(sample_total_tar)
    else:
        test_basic_total_feat.append(sample_total_feat)
        test_basic_total_stat.append(sample_total_stat)
        test_basic_total_tar.append(sample_total_tar)

    index_of_site = obs_years[obs_years['Site_id'] == site_id].index[0]
    for j in range(19):
        year = obs_years.loc[index_of_site, str(j)]
    
        if np.isnan(year):
            break

        i_year = int(year - 1902)

        sample_feat = feat[index, i_year, :]
        sample_stat = static[index, i_year, :]
        sample_tar = targets[index, i_year, :]
        
        #print(sample_feat.shape, obs_pixels.loc[i, "sitesamples"])
        
        if dtype == "train":
            train_basic_feat.append(sample_feat)
            train_basic_stat.append(sample_stat)
            train_basic_tar.append(sample_tar)
        elif dtype == "val":
            val_basic_feat.append(sample_feat)
            val_basic_stat.append(sample_stat)
            val_basic_tar.append(sample_tar)
        else:
            test_basic_feat.append(sample_feat)
            test_basic_stat.append(sample_stat)
            test_basic_tar.append(sample_tar)

train_feat = np.stack(train_basic_feat)
train_stat = np.stack(train_basic_stat)
train_tar = np.stack(train_basic_tar)

val_feat = np.stack(val_basic_feat)
val_stat = np.stack(val_basic_stat)
val_tar = np.stack(val_basic_tar)

test_feat = np.stack(test_basic_feat)
test_stat = np.stack(test_basic_stat)
test_tar = np.stack(test_basic_tar)

train_total_feat = np.stack(train_basic_total_feat)
train_total_stat = np.stack(train_basic_total_stat)
train_total_tar = np.stack(train_basic_total_tar)

val_total_feat = np.stack(val_basic_total_feat)
val_total_stat = np.stack(val_basic_total_stat)
val_total_tar = np.stack(val_basic_total_tar)

test_total_feat = np.stack(test_basic_total_feat)
test_total_stat = np.stack(test_basic_total_stat)
test_total_tar = np.stack(test_basic_total_tar)

# save npy files
np.save(f"{save_path_basic}/basic_train_feat.npy", train_feat)
np.save(f"{save_path_basic}/basic_train_stat.npy", train_stat)
np.save(f"{save_path_basic}/basic_train_tar.npy", train_tar)

np.save(f"{save_path_basic}/basic_val_feat.npy", val_feat)
np.save(f"{save_path_basic}/basic_val_stat.npy", val_stat)
np.save(f"{save_path_basic}/basic_val_tar.npy", val_tar)

np.save(f"{save_path_basic}/basic_test_feat.npy", test_feat)
np.save(f"{save_path_basic}/basic_test_stat.npy", test_stat)
np.save(f"{save_path_basic}/basic_test_tar.npy", test_tar)

np.save(f"{save_path_time_set}/basic_train_total_feat.npy", train_total_feat)
np.save(f"{save_path_time_set}/basic_train_total_stat.npy", train_total_stat)
np.save(f"{save_path_time_set}/basic_train_total_tar.npy", train_total_tar)

np.save(f"{save_path_time_set}/basic_val_total_feat.npy", val_total_feat)
np.save(f"{save_path_time_set}/basic_val_total_stat.npy", val_total_stat)
np.save(f"{save_path_time_set}/basic_val_total_tar.npy", val_total_tar)

np.save(f"{save_path_test_sets}/basic_test_total_feat.npy", test_total_feat)
np.save(f"{save_path_test_sets}/basic_test_total_stat.npy", test_total_stat)
np.save(f"{save_path_test_sets}/basic_test_total_tar.npy", test_total_tar)


#%%
# create test set 2

test2_feat = []
test2_stat = []
test2_tar = []

pixel_test2 = pixels_time_set[(pixels_time_set['dset_type'] == "testset2") | (pixels_time_set['dset_type'] == "test1")]

# loop as above but withour years 
test2_feat = []
test2_stat = []
test2_tar = []

for row in pixel_test2.iterrows():
    index = row[0]

    sample_feat = feat[index, :, :]
    sample_stat = static[index, :, :]
    sample_tar = targets[index, :, :]
    
    test2_feat.append(sample_feat)
    test2_stat.append(sample_stat)
    test2_tar.append(sample_tar)

test2_feat = np.stack(test2_feat)
test2_stat = np.stack(test2_stat)
test2_tar = np.stack(test2_tar)

# save as npy file
np.save(f"{save_path_test_sets}/test2_test_feat.npy", test2_feat)
np.save(f"{save_path_test_sets}/test2_test_stat.npy", test2_stat)
np.save(f"{save_path_test_sets}/test2_test_tar.npy", test2_tar)
# %%
np.random.seed(42)
##############################################
########## select space set ##################
##############################################

train_feat = []
train_stat = []
train_tar = []

val_feat = []
val_stat = []
val_tar = []

pixel_train_space = pixels_space_set[(pixels_space_set['dset_type'] == "train_space")]
pixel_val_space = pixels_space_set[(pixels_space_set['dset_type'] == "val_space")]

for row in pixel_train_space.iterrows():
    index = row[0]

    selected_years = np.random.choice(20, size=8, replace=False)
    # choose 8 random years out of the last 20 years
    sample_feat = feat[index, -20:, :][selected_years]
    sample_stat = static[index, -20:, :][selected_years]
    sample_tar = targets[index, -20:, :][selected_years]

    train_feat.append(sample_feat)
    train_stat.append(sample_stat)
    train_tar.append(sample_tar)

train_feat = np.stack(train_feat)
train_stat = np.stack(train_stat)
train_tar = np.stack(train_tar)

for row in pixel_val_space.iterrows():
    index = row[0]

    selected_years = np.random.choice(20, size=8, replace=False)
    # choose 8 random years out of the last 20 years
    sample_feat = feat[index, -20:, :][selected_years]
    sample_stat = static[index, -20:, :][selected_years]
    sample_tar = targets[index, -20:, :][selected_years]
    
    print(sample_feat.shape)

    val_feat.append(sample_feat)
    val_stat.append(sample_stat)
    val_tar.append(sample_tar)

val_feat = np.stack(val_feat)
val_stat = np.stack(val_stat)
val_tar = np.stack(val_tar)

np.save(f"{save_path_space_set}/space_train_feat.npy", train_feat)
np.save(f"{save_path_space_set}/space_train_stat.npy", train_stat)
np.save(f"{save_path_space_set}/space_train_tar.npy", train_tar)

np.save(f"{save_path_space_set}/space_val_feat.npy", val_feat)
np.save(f"{save_path_space_set}/space_val_stat.npy", val_stat)
np.save(f"{save_path_space_set}/space_val_tar.npy", val_tar)
# direkt die 6 einzelnen Sets aufstelle
# selektiere erst soviele Sites dann die nächsten und so on

# %%
np.random.seed(42)
##############################################
########## select space time set #############
##############################################

train_feat = []
train_stat = []
train_tar = []

val_feat = []
val_stat = []
val_tar = []

pixel_train_spacetime = pixels_spacetime_set[(pixels_spacetime_set['dset_type'] == "train_timespace")]
pixel_val_spacetime = pixels_spacetime_set[(pixels_spacetime_set['dset_type'] == "val_timespace")]

for row in pixel_train_spacetime.iterrows():
    index = row[0]

    sample_feat = feat[index, -35:, :]
    sample_stat = static[index, -35:, :]
    sample_tar = targets[index, -35:, :]

    train_feat.append(sample_feat)
    train_stat.append(sample_stat)
    train_tar.append(sample_tar)

train_feat = np.stack(train_feat)
train_stat = np.stack(train_stat)
train_tar = np.stack(train_tar)

for row in pixel_val_spacetime.iterrows():
    index = row[0]

    selected_years = np.random.choice(20, size=8, replace=False)
    sample_feat = feat[index, -20:, :]
    sample_stat = static[index, -20:, :]
    sample_tar = targets[index, -20:, :]
    
    val_feat.append(sample_feat)
    val_stat.append(sample_stat)
    val_tar.append(sample_tar)

val_feat = np.stack(val_feat)
val_stat = np.stack(val_stat)
val_tar = np.stack(val_tar)


np.save(f"{save_path_timespace_set}/timespace_train_feat.npy", train_feat)
np.save(f"{save_path_timespace_set}/timespace_train_stat.npy", train_stat)
np.save(f"{save_path_timespace_set}/timespace_train_tar.npy", train_tar)

np.save(f"{save_path_timespace_set}/timespace_val_feat.npy", val_feat)
np.save(f"{save_path_timespace_set}/timespace_val_stat.npy", val_stat)
np.save(f"{save_path_timespace_set}/timespace_val_tar.npy", val_tar)
# %%
