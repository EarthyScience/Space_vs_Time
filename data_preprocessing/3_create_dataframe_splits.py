"""
Preprocessing Step 3:

Create dataframes with information for the train/validation/test splits from pixel-level data.

"""

# %%
import numpy as np
import pandas as pd

from pathlib import Path
import os

# %%

DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

pixel_dir = DATA_ROOT / "pixel_information"

# %%
# Load pixel metadata
pixels = pd.read_csv(pixel_dir / "all_pixels_1.csv", index_col=0)

# Load site-to-pixel mapping
fluxnet_jsbach_pixel = pd.read_csv(
    pixel_dir / "fluxnetsite_to_jsbach_pixel.csv",
    index_col=0
)

# Drop specific invalid / unwanted sites (e.g., Italian sites)
fluxnet_jsbach_pixel = (
    fluxnet_jsbach_pixel
    .drop([10, 12])
    .reset_index(drop=True)
)

# Ensure reproducibility
np.random.seed(42)

# Set new column with data type: For all pixels none
pixels["dset_type"] = None
pixels["obs_set"] = False
pixels["obs_siteid"] = None
# set for DE-Hai and Fr-Fon testset1

# Get the coordinates for the sites of interest
sites_of_interest = ["DE-Hai", "FR-Fon"]

for site in sites_of_interest:
    # Filter lat and lon for the current site
    lat = fluxnet_jsbach_pixel[fluxnet_jsbach_pixel['site_id'] == site]["lat"].iloc[0]
    lon = fluxnet_jsbach_pixel[fluxnet_jsbach_pixel['site_id'] == site]["lon"].iloc[0]
    
    # Find the corresponding index in the pixels dataframe
    index_pixel = pixels[(pixels["Latitude"] == lat) & (pixels["Longitude"] == lon)].index
    
    # set corresponding jsbach pixels to test1
    pixels.loc[index_pixel, "dset_type"] = "test1"
    pixels.loc[index_pixel, "obs_set"] = True
    pixels.loc[index_pixel, "obs_siteid"] = site
    
    
    print(f"Indexes for site {site}: {index_pixel.tolist()}")




## train sites to pixel
# set for all pixel in observation set the correct type
train_sites = fluxnet_jsbach_pixel[fluxnet_jsbach_pixel['dset_type'] == "train"]["site_id"]

for site in train_sites:
    # Filter lat and lon for the current site
    lat = fluxnet_jsbach_pixel[fluxnet_jsbach_pixel['site_id'] == site]["lat"].iloc[0]
    lon = fluxnet_jsbach_pixel[fluxnet_jsbach_pixel['site_id'] == site]["lon"].iloc[0]
    
    # Find the corresponding index in the pixels dataframe
    index_pixel = pixels[(pixels["Latitude"] == lat) & (pixels["Longitude"] == lon)].index
    
    if site == "CA-Oas":
        # search for the closest pixel: euclidean distance
        pixels["distance"] = np.sqrt((pixels['Latitude'] - lat)**2 + (pixels['Longitude'] - lon)**2)
        index_pixel = pixels['distance'].idxmin()
        pixels.drop(columns='distance', inplace=True)
    
    print(site, lat, lon, index_pixel)
    # set corresponding jsbach pixels to test1
    pixels.loc[index_pixel, "dset_type"] = "train"
    pixels.loc[index_pixel, "obs_set"] = True
    pixels.loc[index_pixel, "obs_siteid"] = site

# set one more random pixel to train that has obs_set = false
df_filtered = pixels[pixels['obs_set'] == False]
# Randomly select 1 row
selected_indices = df_filtered.sample(n=1, random_state=42).index
# Assign "testset2" to the dset_type column for the selected rows
pixels.loc[selected_indices, 'dset_type'] = "train"
pixels.loc[selected_indices, 'obs_set'] = True
pixels.loc[selected_indices, "obs_siteid"] = "extra_site"

## validation sites to pixel
# set for all pixel in observation set the correct type
val_sites = fluxnet_jsbach_pixel[fluxnet_jsbach_pixel['dset_type'] == "val"]["site_id"]

for site in val_sites:
    # Filter lat and lon for the current site
    lat = fluxnet_jsbach_pixel[fluxnet_jsbach_pixel['site_id'] == site]["lat"].iloc[0]
    lon = fluxnet_jsbach_pixel[fluxnet_jsbach_pixel['site_id'] == site]["lon"].iloc[0]
    
    # Find the corresponding index in the pixels dataframe
    index_pixel = pixels[(pixels["Latitude"] == lat) & (pixels["Longitude"] == lon)].index
    
    # set corresponding jsbach pixels to test1
    pixels.loc[index_pixel, "dset_type"] = "val"
    pixels.loc[index_pixel, "obs_set"] = True
    pixels.loc[index_pixel, "obs_siteid"] = site

### set 28 pixels with obs_set = FALSE as test 2 pixel
# Filter rows where obs_set is False
df_filtered = pixels[pixels['obs_set'] == False]

# Randomly select 28 rows
selected_indices = df_filtered.sample(n=28, random_state=42).index

# Assign "testset2" to the dset_type column for the selected rows
pixels.loc[selected_indices, 'dset_type'] = "testset2"


#%%
# copy pixels dataframe
pixels_spaceset = pixels.copy()
# select 150 training pixels and set to train
spaceset_filtered = pixels_spaceset[(pixels_spaceset['obs_set'] == False) & (pixels_spaceset['dset_type'].isna())]
selected_indices = spaceset_filtered.sample(n=150, random_state=42).index
pixels_spaceset.loc[selected_indices, 'dset_type'] = "train_space"

# select 18 (per step 3) valiadtion pixels and set to val
spaceset_filtered = pixels_spaceset[(pixels_spaceset['obs_set'] == False) & (pixels_spaceset['dset_type'].isna())]
selected_indices = spaceset_filtered.sample(n=18, random_state=42).index
pixels_spaceset.loc[selected_indices, 'dset_type'] = "val_space"


## set space + time set pixels

# copy pixels dataframe 
pixels_spacetimeset = pixels.copy()
# select 34 training pixels and set to train
spacetimeset_filtered = pixels_spacetimeset[(pixels_spacetimeset['obs_set'] == False) & (pixels_spacetimeset['dset_type'].isna())]
selected_indices = spacetimeset_filtered.sample(n=34, random_state=42).index
pixels_spacetimeset.loc[selected_indices, 'dset_type'] = "train_timespace"

# select 7 validation pixels and set to val
spacetimeset_filtered = pixels_spacetimeset[(pixels_spacetimeset['obs_set'] == False) & (pixels_spacetimeset['dset_type'].isna())]
selected_indices = spacetimeset_filtered.sample(n=7, random_state=42).index
pixels_spacetimeset.loc[selected_indices, 'dset_type'] = "val_timespace"


#%%
# save dataframes
pixels.to_csv(pixel_dir / "pixels_timeset.csv")
pixels_spaceset.to_csv(pixel_dir / "pixels_spaceset.csv")
pixels_spacetimeset.to_csv(pixel_dir / "pixels_spacetimeset.csv")




# %%
