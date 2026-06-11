"""
Different model classes for carbonflux ml project
Resnet 18, Resnet 50, Resnet 152, linear model

Version = 1.0
Author = drachti (drachti@bgc-jena.mpg.de)
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision.models import (
    ResNet18_Weights,
)


class Resnet_model18(nn.Module):
    def __init__(self, num_input: int, num_classes: int):
        super(Resnet_model18, self).__init__()
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.body = nn.Sequential(*list(model.children())[:-1])
        self.body[0] = nn.Conv2d(
            num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.head_lay1 = nn.Linear(515, 1024)
        self.head_lay2 = nn.ReLU()
        self.head_lay3 = nn.Linear(1024, out_features=num_classes)

    def forward(
        self, meteo_data: torch.Tensor, static_data: torch.Tensor
    ) -> torch.Tensor:
        body_output = self.body(meteo_data)
        output_size = body_output.shape[0]
        body_output = body_output.reshape([output_size, 512])
        head_input = torch.concat([body_output, static_data], dim=1)
        head_lay1_output = self.head_lay1(head_input)
        head_lay2_output = self.head_lay2(head_lay1_output)
        output = self.head_lay3(head_lay2_output)

        return output


class Model18_IG_sum(nn.Module):
    def __init__(self, num_input: int, num_classes: int):
        super(Model18_IG_sum, self).__init__()
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.body = nn.Sequential(*list(model.children())[:-1])
        self.body[0] = nn.Conv2d(
            num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.head_lay1 = nn.Linear(515, 1024)
        self.head_lay2 = nn.ReLU()
        self.head_lay3 = nn.Linear(1024, out_features=num_classes)

    def forward(
        self, meteo_data: torch.Tensor, static_data: torch.Tensor
    ) -> torch.Tensor:
        body_output = self.body(meteo_data)
        output_size = body_output.shape[0]
        body_output = body_output.reshape([output_size, 512])
        head_input = torch.concat([body_output, static_data], dim=1)
        head_lay1_output = self.head_lay1(head_input)
        head_lay2_output = self.head_lay2(head_lay1_output)
        output = self.head_lay3(head_lay2_output)

        output = torch.reshape(output, (50, 365, 2))
        output = output.sum(dim=1)
        output = torch.squeeze(output)

        return output


class Model18_IG_sum_casestudy(nn.Module):
    def __init__(self, num_input: int, num_classes: int, start_index, end_index):
        super(Model18_IG_sum_casestudy, self).__init__()
        model = models.resnet18(weights=ResNet18_Weights.DEFAULT)
        self.body = nn.Sequential(*list(model.children())[:-1])
        self.body[0] = nn.Conv2d(
            num_input, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False
        )
        self.head_lay1 = nn.Linear(515, 1024)
        self.head_lay2 = nn.ReLU()
        self.head_lay3 = nn.Linear(1024, out_features=num_classes)
        self.start_index = start_index
        self.end_index = end_index

    def forward(
        self, meteo_data: torch.Tensor, static_data: torch.Tensor
    ) -> torch.Tensor:
        body_output = self.body(meteo_data)
        output_size = body_output.shape[0]
        body_output = body_output.reshape([output_size, 512])
        head_input = torch.concat([body_output, static_data], dim=1)
        head_lay1_output = self.head_lay1(head_input)
        head_lay2_output = self.head_lay2(head_lay1_output)
        output = self.head_lay3(head_lay2_output)

        output = torch.reshape(output, (50, 365, 2))
        output = output[:, self.start_index : self.end_index, :].mean(dim=1)
        output = torch.squeeze(output)

        return output


if __name__ == "__main__":
    pass
