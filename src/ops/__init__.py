from src.ops import loss, metrics

metrics_obj_map = {
    "ArcMarginProduct": metrics,
    "AddMarginProduct": metrics,
    "SphereProduct": metrics,
}
loss_obj_map = {"FocalLoss": loss}
