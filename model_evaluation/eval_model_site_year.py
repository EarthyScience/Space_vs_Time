"""
Test the best space and time model (increment 6) on each test gridcell and each test year
"""

# %%
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

from ml_models import Resnet_model18
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm

# %%

model_set = "timespace"

model_code = f"resnet18_{model_set}_1"
n_epochs = 300

model_path = f"trained_models/version1/{model_set}_set"
save_path = f"trained_models/version1/results/model_performance"
save_path_nepsum = f"trained_models/version1/results"
test_data_path = "data/datasets/test_sets"

# init model
device = torch.device("cpu")
num_input = 11
num_output = 365 * 2
model = Resnet_model18(num_input, num_output).to(device)

# total test
test_feat = torch.from_numpy(np.load(f"{test_data_path}/test2_test_feat.npy")).float()
test_stat = torch.from_numpy(np.load(f"{test_data_path}/test2_test_stat.npy")).float()
test_tar = np.load(f"{test_data_path}/test2_test_tar.npy")

# init dataframe
basic_obs_results = pd.DataFrame()

# %%

set_results = pd.DataFrame()

# only load the last model of space/time
k = 5
# create list for dataframe of each model
models_df = []

nep_sum = np.zeros((5, 30, 30))  # save nep sum for each year
nep_target_sum = np.zeros((5, 30, 30))
nep_model = np.zeros((5, 30, 30, 2, 365))
nep_target = np.zeros((5, 30, 30, 2, 365))


for i in range(5):

    # load model weights
    modelinit_path = (
        f"{model_path}/{k+1}_{model_code}_{n_epochs}/init_{i}/m{model_code}.pt"
    )
    model.load_state_dict(torch.load(modelinit_path, map_location=torch.device("cpu")))

    r2_total = pd.DataFrame()

    # loop through all sites
    for j in range(test_feat.shape[0]):

        feat = test_feat[j, -30:, :]
        stat = test_stat[j, -30:, :]
        tar = test_tar[j, -30:, :].transpose(0, 2, 1)

        msites_outputs = []
        msites_yearlyout = []

        m_output = model(feat, stat)
        m_output = torch.reshape(m_output, (30, 365, 2)).transpose(1, 2)
        m_output = m_output.detach().numpy()  # shape now [30, 2, 365]

        moutput_sum = np.sum(m_output, axis=2)
        tar_sum = np.sum(tar, axis=2)

        nep_sum[i, j, :] = moutput_sum[:, 0]
        nep_target_sum[i, j, :] = tar_sum[:, 0]
        nep_model[i, j, :] = m_output
        nep_target[i, j, :] = tar
        ## calculate r^2 for anomalies
        # m_mean_cycle =  m_output.mean(axis=0, keepdims=True)
        # m_anom = m_output - m_mean_cycle
        # tar_mean_cycle =  tar.mean(axis=0, keepdims=True)
        # tar_anom = tar - tar_mean_cycle

        # reshape array for r2 calculations

        # m_anom = m_anom.transpose(1,0,2).reshape([2, 30*365])
        # tar_anom = tar_anom.transpose(1,0,2).reshape([2, 30*365])

        for l in range(30):
            target_sample = tar[l, :]
            m_output_sample = m_output[l, :]

            r2_total.loc[j, l] = 1 / (
                2 - r2_score(target_sample[0, :], m_output_sample[0, :])
            )

        # calculate r^2 values
        m_output = m_output.transpose(1, 0, 2).reshape([2, 30 * 365])
        tar = tar.transpose(1, 0, 2).reshape([2, 30 * 365])
        r2_total.loc[j, "site_mean"] = 1 / (2 - r2_score(tar[0, :], m_output[0, :]))

    models_df.append(r2_total)

# calculate the average over each model
average_df = sum(models_df) / len(models_df)
# calculate std
# Stack DataFrames to calculate standard deviation
stacked = np.stack([df.values for df in models_df], axis=0)
# Calculate standard deviation along the 0th axis (over all DataFrames)
std_dev = np.std(stacked, axis=0)
# Convert the result back to a DataFrame
std_dev_df = pd.DataFrame(std_dev, columns=r2_total.columns, index=r2_total.index)

# calculate mean over models for each nep sum
nep_sum_mean = nep_sum.mean(axis=0)
nep_target_sum_mean = nep_target_sum.mean(axis=0)
# save nep sums
np.save(f"{save_path_nepsum}/nep_sum_{model_set}_siteyear.npy", nep_sum_mean)

# save nep model predictions
nep_model_mean = nep_model.mean(axis=0)
nep_target_mean = nep_target.mean(axis=0)

np.save(f"{save_path_nepsum}/nep_model_{model_set}_siteyear.npy", nep_model_mean)
np.save(f"{save_path_nepsum}/nep_target_siteyear.npy", nep_target_mean)


# np.save(f"{save_path_nepsum}/nep_sum_target_siteyear.npy", nep_target_sum_mean)
# %%
# save dataframes
average_df.to_csv(f"{save_path}/{model_set}_perf_r2_site_year.csv")
std_dev_df.to_csv(f"{save_path}/{model_set}_perf_r2std_site_year.csv")
# %%
