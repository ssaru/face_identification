import torch
import torch.nn as nn


class NormHead(nn.Module):
    def __init__(
        self,
        in_features: int,
        num_classes: int,
        w_decay: float = 5e-4,
    ):
        super(NormHead, self).__init__()
        self.w_deacy = w_decay
        self.input = nn.Linear(
            in_features=in_features,
            out_features=num_classes,
            bias=True,
        )

    def forward(self, x: torch.Tensor):
        x = self.input(x)
        return x
