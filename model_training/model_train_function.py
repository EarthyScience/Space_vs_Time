"""
Model pytorch basic train function
Version = 1.0
Author = drachti (drachti@bgc-jena.mpg.de)
"""

#%%
import copy
import time
import numpy as np
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

# %%

class Identity:
    """
    Basic class for the preprocessor. Does nothing

    Args:
        data (torch.Tensor): The dataset data
    """

    def __init__(self, data: torch.Tensor):
        """initilizes the identiticlass doing nothing"""

    def process(self, sample: torch.Tensor) -> (torch.Tensor):
        """
        The forward method for preprocessing. Does nothing
        Args:
            sample (torch.Tensor): The features of the sample that should be
                preprocessed

        Returns:
            torch.Tensor: The preprocessed features of the sample
        """
        return sample

    def deprocess(self, sample: torch.Tensor) -> (torch.Tensor):
        """
        The backward method for preprocessing that recovers the original values.
        Does nothing
        Args:
            sample (torch.Tensor): The preprocessed features of the sample that
                should be recovered.
        Returns:
            torch.Tensor: The recovered features of the sample
        """
        return sample


class SimpleDataset(torch.utils.data.Dataset):
    """
    A simple PyTorch dataset class that loads data onto the given device.

    Args:
    - data (List[List[float]]): A list of data samples.
    - device (str): The device on which the data samples will be loaded.

    Returns:
    - A PyTorch Dataset object that can be used with DataLoader.
    """

    def __init__(
        self,
        meteo: np.array,
        static: np.array,
        targets: np.array,
        device: str or None = None,
        preprocessing: str or Identity or None = None,
    ):

        # Set device in case the device is not set already
        if device is None:
            self.device = "cpu"
            if torch.cuda.is_available():
                self.device = "cuda"

            print(f"device set to {self.device}")

        self.meteo = meteo
        self.static = static
        self.targets = targets
        self.device = device

    def __len__(self) -> int:
        """
        Returns the number of samples in the dataset.
        """
        return self.meteo.shape[0]

    def __getitem__(self, idx: int) -> torch.Tensor:
        """
        Gets the data sample at the given index and loads it onto the device.

        Args:
        - idx (int): The index of the sample to retrieve.

        Returns:
        - A PyTorch tensor of the data sample loaded onto the given device.
        """
        meteo = torch.tensor(self.meteo[idx, :, :, :], device=self.device).float()
        static = torch.tensor(self.static[idx, :], device=self.device).float()
        targets = torch.tensor(self.targets[idx, :, :], device=self.device).float()

        return (
            meteo,
            static,
            targets,
        )

def plot_loss_one_epoch(loss_data, n_epochs, date, path):
    fig = plt.figure(figsize=(10, 6))
    ax = fig.add_subplot()

    ax.plot(np.abs(loss_data[0]), label="Train Loss")
    ax.plot(np.abs(loss_data[1]), label="Val Loss")
    ax.set_ylabel("epoch loss (MSE)")
    ax.set_xlabel("Runs (Epochs)")
    ax.set_title(f"Epoch Loss ({n_epochs} Epochs)")
    ax.legend(loc=0)
    ax.set_yscale("log")
    fig.savefig(f"{path}/epoch_loss.png", dpi=300)
    return


#%%


def trainings_loop(
    optimizer,
    model,
    loss_fn,
    scheduler,
    n_epochs,
    date,
    path,
    batch_size,
    train_loader,
    val_loader,
):

    since = time.time()
    model_pre = copy.deepcopy(model.state_dict())
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 10000

    train_loss = []
    val_loss = []

    model.load_state_dict(model_pre)

    for epoch in range(n_epochs):

        # Each Epoch has a training and validation phase
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                data_set = train_loader

            else:
                model.eval()
                data_set = val_loader
            running_loss = 0.0

            # Iterate over list of tensors
            for batch in data_set:
                meteo_input = batch[0]
                static_input = batch[1]
                target_flux = batch[2]
                batch_rsize = meteo_input.shape[0]
                # if phase == "val":
                #     print(torch.isnan(meteo_input).sum())
                #     print("target", torch.isnan(target_flux).sum())

                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(meteo_input, static_input)
                    outputs = torch.reshape(outputs, (target_flux.shape))
                    loss_mse = loss_fn(outputs, target_flux)
                    #loss_ylsum_mse = loss_fn(torch.sum(outputs, dim=2), torch.sum(target_flux, dim=2))
                    # print(torch.sum(outputs, dim=2).shape, torch.sum(target_flux, dim=2).shape)
                    # print(torch.sum(outputs, dim=2), torch.sum(target_flux, dim=2))
                    # print(loss_mse, loss_ylsum_mse)
                    # loss function
                    loss = loss_mse# + loss_ylsum_mse/365

                    # backward + optimize only if in training phase
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * batch_rsize

            epoch_loss = running_loss / len(data_set)

            if phase == "train":
                train_loss.append(epoch_loss)
                # learning rate scheduler
                scheduler.step()
            else:
                val_loss.append(epoch_loss)

            if epoch <= 3 or epoch % 10 == 0 or epoch == n_epochs - 1:
                print(f"Epoch  {epoch}, \n {phase} Loss: {epoch_loss:.4f}")

            # deep copy the model
            if phase == "val" and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_model_wts = copy.deepcopy(model.state_dict())


    print(f"Training complete")

    model.load_state_dict(best_model_wts)
    torch.save(
        model.state_dict(),
        f"{path}/m{date}.pt",
    )
    loss_data = np.array(
        [
            train_loss[-epoch:],
            val_loss[-epoch:],
        ]
    )
    plot_loss_one_epoch(loss_data, n_epochs, date, path)

    time_elapsed = time.time() - since

    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")

    loss_data = np.array([train_loss, val_loss])

    # load best model weights
    return loss_data