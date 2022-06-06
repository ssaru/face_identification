from turtle import forward

import torch
import torch.nn as nn


class Outputlayer(nn.Module):
    def __init__(self, in_features: int, w_decay: float):
        super(Outputlayer, self).__init__()
        self.layers = nn.Sequential(
            nn.BatchNorm2d(),
            nn.Dropout(p=0.5),
            nn.Flatten(),
            nn.Linear(in_features=in_features),
            nn.BatchNorm2d(),
        )

    def forward(self, x):
        return self.layers(x)
