from torch.nn import CrossEntropyLoss

from .focal_loss import FocalLoss

__all__ = ["FocalLoss", "CrossEntropyLoss"]
