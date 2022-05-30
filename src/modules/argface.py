from typing import Dict

import torch.nn as nn

from src.networks.backbone import model_obj_map
from src.ops import loss_obj_map, metrics, metrics_obj_map

available_models = list(model_obj_map.keys())
available_metrics = list(metrics_obj_map.keys())
available_loss = list(loss_obj_map.keys())


class ArcFace(nn.Module):
    def __init__(self, backbone: str, metric: Dict, loss: Dict):
        """
        Args:
            backbone (str): {'name': ..., 'params': {...}}
            metric (Dict): {'name': ..., 'params': {...}}
            loss (Dict): {'name': ..., 'params': {...}}
        Returns:
            (obj)
        """
        backbone_name = backbone.get("name")
        backbone_params = backbone.get("params")

        metric_name = metric.get("name")
        metric_params = metric.get("params")

        loss_name = loss.get("name")
        loss_params = loss.get("params")

        if backbone_name not in available_models:
            raise RuntimeError(f"Not Supported Backbone network: '{backbone_name}'")

        if metric_name not in available_metrics:
            raise RuntimeError(f"Not Supported Metric: '{metric_name}'")

        if loss_name not in available_loss:
            raise RuntimeError(f"Not Supported Loss: '{loss_name}'")

        load_backbone_func = getattr(model_obj_map.get(backbone_name), backbone_name)
        metric_cls = getattr(metrics_obj_map.get(metric_name), metric_name)
        loss_cls = getattr(loss_obj_map.get(loss_name), loss_name)

        self.backbone = load_backbone_func(*backbone_params)
        self.metric = metric_cls(*metric_params)
        self.loss = loss_cls(*loss_params)

    def forward(self, x):
        x = self.backbone(x)
        x = self.metric(x)

        return x
