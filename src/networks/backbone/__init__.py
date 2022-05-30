from src.networks.backbone.pretrained import classification

model_obj_map = {
    "resnet18": classification,
    "resnet34": classification,
    "resnet50": classification,
    "resnet101": classification,
    "resnet152": classification,
    "resnext50_32x4d": classification,
    "resnext101_32x8d": classification,
    "wide_resnet50_2": classification,
    "wide_resnet101_2": classification,
}
