"""
Train model on different data subsets using GPU.
"""

# %%
from pathlib import Path

import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from ml_models import Resnet_model18
from model_train_function import trainings_loop, SimpleDataset

# %%

if __name__ == "__main__":

    # Train on datasets with different sizes
    # num_addyr = 20
    # size_name = f"obs_{num_addyr}yr"

    # Set parameters
    model_set = "space"
    acronym = f"resnet18_{model_set}_1"

    n_epochs = 300
    lr = np.pi * 1e-4
    device = "cuda"
    batch_size = 4

    num_input = 11
    num_output = 365 * 2

    # Set GPU device
    device = torch.device(device)

    # Paths
    project_root = Path(__file__).resolve().parents[1]

    data_path = (
        project_root
        / "data"
        / "jsbach_space_time_splits"
        / "datasets"
        / f"{model_set}_set"
    )

    # train first basic set and then build loops over each sample

    for i in range(6):
        step = i + 1
        train_meteo = np.load(f"{data_path}/{model_set}_{step}_train_feat.npy")
        train_static = np.load(f"{data_path}/{model_set}_{step}_train_stat.npy")
        train_targets = np.load(f"{data_path}/{model_set}_{step}_train_tar.npy")

        val_meteo = np.load(f"{data_path}/{model_set}_{step}_val_feat_sites.npy")
        val_static = np.load(f"{data_path}/{model_set}_{step}_val_stat_sites.npy")
        val_targets = np.load(f"{data_path}/{model_set}_{step}_val_tar_sites.npy")

        # for each model five different initialisations
        for j in range(5):

            seed = 42 + j  # Generate unique seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)

            train_dataset = SimpleDataset(
                train_meteo, train_static, train_targets, device=device
            )
            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=lambda _: torch.manual_seed(seed),
            )

            val_dataset = SimpleDataset(
                val_meteo, val_static, val_targets, device=device
            )
            val_dataloader = torch.utils.data.DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=True,
                worker_init_fn=lambda _: torch.manual_seed(seed),
            )

            # init model
            model = Resnet_model18(num_input, num_output).to(device)

            optimizer = optim.AdamW(
                model.parameters(), lr=lr, weight_decay=0.1
            )  # Optimizer
            loss_fn = nn.MSELoss()  # Loss function
            scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, T_0=75
            )  # Learning rate scheduler

            # path
            path = f"trained_models/version1/{model_set}_set/{i+1}_{acronym}_{n_epochs}/init_{j}"
            os.makedirs(path, exist_ok=True)

            # train function
            print(path)
            loss_data = trainings_loop(
                optimizer=optimizer,
                model=model,
                loss_fn=loss_fn,
                scheduler=scheduler,
                train_loader=train_dataloader,
                val_loader=val_dataloader,
                n_epochs=n_epochs,
                date=acronym,
                batch_size=batch_size,
                path=path,
            )

            np.savetxt(f"{path}/m{acronym}_E_{n_epochs}.csv", loss_data, delimiter=";")
