from typing import Dict

import torch.nn as nn

from src.core.layers import loss, metrics
from src.core.networks import backbone


class ArcFace(nn.Module):
    def __init__(self, backbone_conf: Dict, metric_conf: Dict, loss_conf: Dict):
        """
        Args:
            backbone (str): {'name': ..., 'params': {...}}
            metric (Dict): {'name': ..., 'params': {...}}
            loss (Dict): {'name': ..., 'params': {...}}
        Returns:
            (obj)
        """
        super(ArcFace, self).__init__()
        backbone_name = backbone_conf.get("name")

        metric_name = metric_conf.get("name")
        metric_params = metric_conf.get("params")

        loss_name = loss_conf.get("name")
        loss_params = loss_conf.get("params")

        if backbone_name not in backbone.__all__:
            raise RuntimeError(f"Not Supported Backbone network: '{backbone_name}'")

        if metric_name not in metrics.__all__:
            raise RuntimeError(f"Not Supported Metric: '{metric_name}'")

        if loss_name not in loss.__all__:
            raise RuntimeError(f"Not Supported Loss: '{loss_name}'")

        self.backbone = getattr(backbone, backbone_name)()
        self.metric = getattr(metrics, metric_name)(**metric_params)
        self.loss = getattr(loss, loss_name)(**loss_params)

    def forward(self, x):
        x = self.backbone(x)
        x = self.metric(x)
        return x
