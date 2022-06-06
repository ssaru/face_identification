import torch
import torch.nn as nn


class ArcHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        margin: float = 0.5,
        logist_scale: int = 64,
    ):
        super(ArcHead, self).__init__()
        self.in_features = in_features
        self.num_classes = num_classes
        self.margin = margin
        self.logist_scale = logist_scale

    def forward(self, x: torch.Tensor):

        return x
