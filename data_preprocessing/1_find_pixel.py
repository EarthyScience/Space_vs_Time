"""
Preprocessing Step 1:

Find forest pixels from JS-Bach in Europe and east NA for model training, validation and testing.

"""

# %%
from pathlib import Path
import os

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import pandas as pd

# %%
# Root data directory (set externally for reproducibility)
DATA_ROOT = Path(os.environ.get("DATA_ROOT", "data"))

# Load tree fraction data
tfrac_path = DATA_ROOT / "treeFrac.nc"

tree_frac = xr.open_mfdataset(tfrac_path)

# save directory
pixel_dir = DATA_ROOT / "pixel_information"

# %%
# select central europe
lat_min, lat_max = 45, 63
lon_min, lon_max = 0, 30

tree_frac_eu = tree_frac.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
tree_frac_eu = tree_frac_eu.groupby("time.year").mean(dim="time")
mean_tree_frac = tree_frac_eu.mean(dim="year")
forest_pix_eu = tree_frac_eu.treeFrac.where(tree_frac_eu.treeFrac > 10, np.nan)

# Check for pixels where all values across time are non-NaN
valid_pixels_mask = ~np.isnan(forest_pix_eu).all(dim="year")
valid_pixels_mask_values = valid_pixels_mask.values

# %%
pixel_info = []

# Iterate through all pixels
from tqdm import tqdm

for lat_idx, lat in tqdm(
    enumerate(valid_pixels_mask.lat.values),
    total=len(valid_pixels_mask.lat.values),
    desc="Processing latitudes",
):
    for lon_idx, lon in enumerate(valid_pixels_mask.lon.values):
        # Check if the pixel is valid (no NaN across all years)
        if valid_pixels_mask_values[lat_idx, lon_idx]:
            # Append the coordinates to the list
            mean_tree_frac_pix = (
                tree_frac_eu.treeFrac.sel(lat=lat, lon=lon).mean(dim="year").values
            )
            pixel_info.append(
                {
                    "Latitude": lat,
                    "Longitude": lon,
                    "region": "EU",
                    "mean_TreeFrac": mean_tree_frac_pix,
                }
            )

# Convert the list of dictionaries to a pandas DataFrame
eu_pix = pd.DataFrame(pixel_info)

# Output the DataFrame
print(eu_pix)

# %%
# select east NA
lat_min, lat_max = 35, 50
lon_min, lon_max = 360 - 95, 360 - 60

tree_frac_na = tree_frac.sel(lon=slice(lon_min, lon_max), lat=slice(lat_min, lat_max))
tree_frac_na = tree_frac_na.groupby("time.year").mean(dim="time")
forest_pix_na = tree_frac_na.treeFrac.where(tree_frac_na.treeFrac > 10, np.nan)

# Check for pixels where all values across time are non-NaN
valid_pixels_mask = ~np.isnan(forest_pix_na).all(dim="year")
valid_pixels_mask_values = valid_pixels_mask.values

# %%
pixel_info = []

# Iterate through all pixels
for lat_idx, lat in tqdm(
    enumerate(valid_pixels_mask.lat.values),
    total=len(valid_pixels_mask.lat.values),
    desc="Processing latitudes",
):
    for lon_idx, lon in enumerate(valid_pixels_mask.lon.values):
        # Check if the pixel is valid (no NaN across all years)
        if valid_pixels_mask_values[lat_idx, lon_idx]:
            # Append the coordinates to the list
            mean_tree_frac_pix = (
                tree_frac_na.treeFrac.sel(lat=lat, lon=lon).mean(dim="year").values
            )
            pixel_info.append(
                {
                    "Latitude": lat,
                    "Longitude": lon,
                    "region": "NA",
                    "mean_TreeFrac": mean_tree_frac_pix,
                }
            )

# Convert the list of dictionaries to a pandas DataFrame
na_pix = pd.DataFrame(pixel_info)
# %%

na_pix
# %%

all_pixels_df = pd.concat([eu_pix, na_pix]).reset_index(drop=True)

all_pixels_df.to_csv(pixel_dir / "all_pixels_1.csv")
# %%
