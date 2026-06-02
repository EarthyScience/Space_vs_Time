"""
Test the basic obs models with test data
"""
#%%
import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

from scipy.stats import linregress
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from scipy.stats import linregress

from ml_models import Resnet_model18
from matplotlib.colors import ListedColormap
from matplotlib.colors import LogNorm


#%%

model_set = "timespace"

model_code = f"resnet18_{model_set}_1"
n_epochs = 300

model_path = f"../trained_models/version1/{model_set}_set"
save_path = f"../trained_models/version1/results/model_performance"
test_data_path = "../data/datasets/test_sets"

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

#%%

set_results = pd.DataFrame()

# loop through all 6 stacks
for k in range(6):

    # set up dataframes
    r2_total = pd.DataFrame()
    r2_anom = pd.DataFrame()
    r2_iav = pd.DataFrame()
    r2_iav_2 = pd.DataFrame()
    trend_iav_mout = pd.DataFrame()
    trend_iav_tar = pd.DataFrame()
    trend_iav_mout_p = pd.DataFrame()
    trend_iav_tar_p = pd.DataFrame()
    mse = pd.DataFrame()

    for i in range(5):

        # load model weights
        modelinit_path = f"{model_path}/{k+1}_{model_code}_{n_epochs}/init_{i}/m{model_code}.pt"
        model.load_state_dict(torch.load(modelinit_path, map_location=torch.device('cpu')))

        # loop through all sites
        for j in range(test_feat.shape[0]):

            feat = test_feat[j, -30:, :]
            stat = test_stat[j, -30:, :]
            tar = test_tar[j, -30:, :].transpose(0,2,1)

            msites_outputs = []
            msites_yearlyout = []

            m_output = model(feat, stat)
            m_output = torch.reshape(m_output, (30,365,2)).transpose(1,2)
            m_output = m_output.detach().numpy() # shape now [30, 2, 365]
            
            moutput_sum = np.sum(m_output, axis=2)
            tar_sum = np.sum(tar, axis=2)
        
            ## calculate r^2 for anomalies
            m_mean_cycle =  m_output.mean(axis=0, keepdims=True)
            m_anom = m_output - m_mean_cycle
            tar_mean_cycle =  tar.mean(axis=0, keepdims=True)
            tar_anom = tar - tar_mean_cycle

            # reshape array for r2 calculations
            m_output = m_output.transpose(1,0,2).reshape([2, 30*365])
            tar = tar.transpose(1,0,2).reshape([2, 30*365])
            m_anom = m_anom.transpose(1,0,2).reshape([2, 30*365])
            tar_anom = tar_anom.transpose(1,0,2).reshape([2, 30*365])

            # calculate r^2 values
            r2_total.loc[j,i] = r2_score(tar[0,:], m_output[0,:])
            r2_anom.loc[j,i] = r2_score(tar_anom[0,:], m_anom[0,:])
            r2_iav.loc[j,i] = r2_score(tar_sum[:,0], moutput_sum[:,0])
            mse.loc[j,i] = mean_squared_error(tar[0,:], m_output[0,:])
            
            # calcualte r^2 on iav without trend
            x_vals = np.arange(0,30)
            slope, intercept, _, _, pvalue = linregress(x_vals, moutput_sum[:,0])
            slope_tar, intercept_tar, _, _, pvalue_tar = linregress(x_vals, tar_sum[:,0])
            trend = slope * x_vals + intercept
            trend_tar = slope_tar * x_vals + intercept_tar
            # remove trend
            r2_iav_2.loc[j,i] = r2_score(tar_sum[:,0] - trend, moutput_sum[:,0] - trend_tar)
            # save trend
            trend_iav_mout.loc[j,i] = slope
            trend_iav_tar.loc[j,i] = slope_tar
            trend_iav_mout_p.loc[j,i] = pvalue
            trend_iav_tar_p.loc[j,i] = pvalue_tar

    set_results.loc[k,"r2_total"] = r2_total.mean(axis=0).mean()
    set_results.loc[k,"r2_total_std"] = r2_total.mean(axis=0).std()
    set_results.loc[k,"r2_anom"] = r2_anom.mean(axis=0).mean()
    set_results.loc[k,"r2_anom_std"] = r2_anom.mean(axis=0).std()
    set_results.loc[k,"r2_iav"] = r2_iav.mean(axis=0).mean()
    set_results.loc[k,"r2_iav_std"] = r2_iav.mean(axis=0).std()
    set_results.loc[k,"mse"] = mse.mean(axis=0).mean()
    set_results.loc[k,"mse_std"] = mse.mean(axis=0).std()
    set_results.loc[k,"r2_iav_2"] = r2_iav_2.mean(axis=0).mean()
    set_results.loc[k,"r2_iav_2_std"] = r2_iav_2.mean(axis=0).std()
    set_results.loc[k,"trend_mout"] = trend_iav_mout.mean(axis=0).mean()
    set_results.loc[k,"trend_mout_std"] = trend_iav_mout.mean(axis=0).std()
    set_results.loc[k,"trend_mout_p"] = trend_iav_mout.mean(axis=0).mean()
    set_results.loc[k,"trend_tar"] = trend_iav_tar.mean(axis=0).mean()
    set_results.loc[k,"trend_tar_std"] = trend_iav_tar.mean(axis=0).std()
    set_results.loc[k,"trend_tar_p"] = trend_iav_tar.mean(axis=0).mean()
    
set_results.loc["mean",:] = set_results.mean(axis=0)
set_results.loc["std",:] = set_results.std(axis=0)

set_results.to_csv(f"{save_path}/{model_set}_perfomance.csv")

# save also performance per site (r^2_iav, r^2_anom)
r2_iav.to_csv(f"{save_path}/{model_set}_r2iav_site.csv")
r2_iav_2.to_csv(f"{save_path}/{model_set}_r2iav_2_site.csv")
r2_anom.to_csv(f"{save_path}/{model_set}_r2anom_site.csv")
# %%
