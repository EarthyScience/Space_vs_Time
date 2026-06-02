"""
Preprocessing Step 2:

1. Extract pixel data from nc files and stack in a numpy array
2. Normalize the data
3. Wavelet Transform the data

"""
# %%
import os
from pathlib import Path

import numpy as np
import xarray as xr
import pandas as pd

from wavelet_transform import WaveletTransform

# %%

# Root JSBACH dataset directory
JSBACH_DATA = Path(os.environ.get("DATA_ROOT", "js_bach_data"))

# Load pixel metadata
pixels = pd.read_csv("../data/pixel_information/all_pixels_1.csv", index_col=0)

# Meteorological forcing data
meteo_input = xr.open_mfdataset(
    JSBACH_DATA / "*.nc",
    drop_variables=["lat_bnds", "time_bnds"]
)

fapar = xr.open_dataset(JSBACH_DATA / "fapar.nc")
ba    = xr.open_dataset(JSBACH_DATA / "burntArea.nc")
snw   = xr.open_dataset(JSBACH_DATA / "snw.nc")
tas   = xr.open_dataset(JSBACH_DATA / "tas.nc")
hurs  = xr.open_dataset(JSBACH_DATA / "hurs.nc")
sm    = xr.open_dataset(JSBACH_DATA / "mrso.nc")

co2 = xr.open_dataset(JSBACH_DATA / "global_co2_annual_1700_2021.nc")


# Target variables
tar_nep = xr.open_dataset(JSBACH_DATA / "nep.nc")
tar_hfls = xr.open_dataset(JSBACH_DATA / "hfls.nc")

# Remove leap-day entries
tar_nep = tar_nep.where(
    ~((tar_nep.time.dt.month == 2) & (tar_nep.time.dt.day == 29)),
    drop=True
)

tar_hfls = tar_hfls.where(
    ~((tar_hfls.time.dt.month == 2) & (tar_hfls.time.dt.day == 29)),
    drop=True
)

# ---------------------------------------------------------------------
# Preprocessing outputs
# ---------------------------------------------------------------------
save_path_ppsteps = JSBACH_DATA / "pre_processing_steps"
save_path_final = JSBACH_DATA
# loop through all pixels and load data into numpy array

variables = ["ta", "precip", "hurs", "vpd", "sm", "snw", "shortwave", "longwave", "ba", "fapar", "rand"]


def extract_data(dataset, lat, lon):
    dataset = dataset.where(~((dataset.time.dt.month == 2) & (dataset.time.dt.day == 29)), drop=True)
    ex_data = dataset.sel(lat=lat, lon=lon).values
    ex_data_rsh = np.reshape(ex_data, (121, 365))
    return ex_data_rsh

def extract_meteodata(dataset, lat, lon):
    """
    Extract data for meteo input file
    Cut of the last value for each year
    """
    ex_data = dataset.sel(lat=lat, lon=lon, method="nearest").values
    ex_data_rsh = np.reshape(ex_data, (121, 366))
    ex_data_fin = ex_data_rsh[:,0:365]
    return ex_data_fin

def calc_vpd(rh_data, ta_data):
    """
    Calculate RH from vpd and ta data
    """
    # Buck equation
    # calculate kelvin to celcius
    ta_data_c = ta_data - 273.15
    e_s = 611.21 * np.exp((18.678 - ta_data_c/234.5)*(ta_data_c/(257.14 + ta_data_c)))
    vpd = e_s * (1 - rh_data/100)

    return vpd

#%%

feature_data = []
stat_data =  []
target_data = []

for i in range(pixels.shape[0]):

    print(i)

    data_pred = np.zeros([121, 11, 365])
    data_stat = np.zeros([121, 3])
    data_tar = np.zeros([120, 365, 2])

    lat = pixels.loc[i, "Latitude"]
    lon = pixels.loc[i, "Longitude"]

    # nep_values = tas.tas.sel(lat=lat, lon=lon).values
    data_pred[:, 0, :] = extract_data(tas.tas, lat, lon)
    data_pred[:, 1, :] = extract_meteodata(meteo_input.precip, lat, lon)
    data_pred[:, 2, :] = extract_data(hurs.hurs, lat, lon)
    
    data_pred[:, 4, :] = extract_data(sm.mrso, lat, lon)
    data_pred[:, 5, :] = extract_data(snw.snw, lat, lon)
    data_pred[:, 6, :] = extract_meteodata(meteo_input.shortwave, lat, lon)
    data_pred[:, 7, :] = extract_meteodata(meteo_input.longwave, lat, lon)
    data_pred[:, 8, :] = extract_data(ba.burntArea, lat, lon)
    data_pred[:, 9, :] = extract_data(fapar.fapar, lat, lon)

    # calculate VPD
    data_pred[:, 3, :] = calc_vpd(data_pred[:, 2, :], data_pred[:, 0, :])
    # load co2
    data_stat[:,2] = co2.CO2.sel(time=slice(1901, 2021), lat=0, lon=360)

    # calculate mean temperature
    data_stat[:40,0] = np.mean(data_pred[:40, 0, :])
    data_stat[40:80,0] = np.mean(data_pred[40:80, 0, :])
    data_stat[80:,0] = np.mean(data_pred[80, 0, :])
    # calculate mean precipitation
    data_stat[:40,1] = np.mean(data_pred[:40, 1, :])
    data_stat[40:80,1] = np.mean(data_pred[40:80, 1, :])
    data_stat[80:,1] = np.mean(data_pred[80, 1, :])

    # load target data
    nep_values = tar_nep.nep.sel(lat=lat, lon=lon, time=slice("1902-01-01", "2021-12-31")).values
    data_tar[:, :, 0] = np.reshape(nep_values, (120, 365))

    hfls_values = tar_hfls.hfls.sel(lat=lat, lon=lon, time=slice("1902-01-01", "2021-12-31")).values
    data_tar[:, :, 1] = np.reshape(hfls_values, (120, 365))

    feature_data.append(data_pred)
    stat_data.append(data_stat)
    target_data.append(data_tar)


np_features = np.stack(feature_data)
np_features = np_features.astype(dtype="float32")

np_stat = np.stack(stat_data)
np_stat = np_stat.astype(dtype="float32")

np_target = np.stack(target_data)
np_target = np_target.astype(dtype="float32")
#%%
# save npy (without normalisation and wavelet transform)
np.save(f"{save_path_ppsteps}/pred_orig.npy", np_features)
np.save(f"{save_path_ppsteps}/stat_orig.npy", np_stat)
np.save(f"{save_path_ppsteps}/tar_orig.npy", np_target)

print("Saved files: pred_orig")

###################################
####### Wavelet transform data ####
###################################

# load wavelet transformer:
wavelet_transform = WaveletTransform(64)

### train data

wt_feat_stack = []
stat_stack = []

for i in range(pixels.shape[0]): # number of pixels  

    data_wt_feat = np.zeros([120, 11, 64, 730])
    data_stat = np.zeros([120, 3])
    print(i)
    for j in range(120): # number of years

        for k in range(11):
            if k == 10:
                par_random = 1 + 0.1 *np.random.randn(730)
                data = np.cumprod(par_random)
            elif j == 0 and k == 9:
                # first year first day value of fApar is nan value
                fill_val = np_features[0,0,9,2]
                data = np_features[i,j:j+2,k,:]
                data = np.nan_to_num(data, nan=fill_val) 
                data = np.reshape(data, (730))
            else:
                data = np_features[i,j:j+2,k,:]
                data = np.reshape(data, (730))

            data_wt_feat[j,k,:,:] = wavelet_transform(data)
        # set stat data correctly (First year is removed)
        data_stat[j,:] = np_stat[i,j+1,:]

    wt_feat_stack.append(data_wt_feat)
    stat_stack.append(data_stat)

wt_feat = np.stack(wt_feat_stack, axis=0)
stat_feat = np.stack(stat_stack, axis=0)

np.save(f"{save_path_ppsteps}/pred_wt.npy", wt_feat)
np.save(f"{save_path_ppsteps}/stat_wt.npy", stat_feat)

        
# save data

# normalize data

feat_max = np.max(wt_feat, axis=(0, 1, 3, 4), keepdims=True)
feat_min = np.min(wt_feat, axis=(0, 1, 3, 4), keepdims=True)

# save min max values for denormalisation as npy
np.save(f"{save_path_ppsteps}/feat_max_normval.npy", feat_max)
np.save(f"{save_path_ppsteps}/feat_min_normval.npy", feat_min)

print("feat max",feat_max.shape)

def min_max_norm(data, max, min):
    """ small function do normalize feat and target data """
    data = data - min
    data = data / (max - min)

    return data

wt_feat_norm = min_max_norm(wt_feat, feat_max, feat_min)

# normalize statics

mat_max = np.max(stat_feat[:,:,0])
mat_min = np.min(stat_feat[:,:,0])
map_max = np.max(stat_feat[:,:,1])
map_min = np.min(stat_feat[:,:,1])
co2_max = np.max(stat_feat[:,:,2])
co2_min = np.min(stat_feat[:,:,2])

def min_max_norm2(data, max, min):
    """ small function to normalize stat data """
    data[:, :,0] = data[:, :,0] - min[0]
    data[:, :,0] = data[:, :,0] / (max[0] - min[0])

    data[:, :,1] = data[:, :,1] - min[1]
    data[:, :,1] = data[:, :,1] / (max[1] - min[1])

    data[:, :,2] = data[:, :,2] - min[2]
    data[:, :,2] = data[:, :,2] / (max[2] - min[2])

    return data

stat_feat_norm = min_max_norm2(stat_feat, np.array([mat_max, map_max, co2_max]), np.array([mat_min, map_min, co2_min]))

## normalize targets
tar_max = np.max(np_target, axis=(0,1,2), keepdims=True)
tar_min = np.min(np_target, axis=(0,1,2), keepdims=True)
print("max", tar_max, "min", tar_min)
target_norm = min_max_norm(np_target, tar_max, tar_min)

print("tar max",tar_max.shape)

np.save(f"{save_path_final}/allfeat_wt_norm.npy", wt_feat_norm)
np.save(f"{save_path_final}/all_stat_norm.npy", stat_feat_norm)
np.save(f"{save_path_final}/all_tar_norm.npy", target_norm)
# %%