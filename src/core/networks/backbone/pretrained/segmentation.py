import torch
from torchvision.models import segmentation

fcn_resnet50 = segmentation.fcn_resnet50(pretrained=True)
fcn_resnet50 = torch.nn.Sequential(*(list(fcn_resnet50.children())[:-2]))

fcn_resnet101 = segmentation.fcn_resnet101(pretrained=True)
fcn_resnet101 = torch.nn.Sequential(*(list(fcn_resnet101.children())[:-2]))

deeplabv3_resnet50 = segmentation.deeplabv3_resnet50(pretrained=True)
deeplabv3_resnet50 = torch.nn.Sequential(*(list(deeplabv3_resnet50.children())[:-2]))

deeplabv3_mobilenet_v3_large = segmentation.deeplabv3_mobilenet_v3_large(
    pretrained=True
)
deeplabv3_mobilenet_v3_large = torch.nn.Sequential(
    *(list(deeplabv3_mobilenet_v3_large.children())[:-2])
)

lraspp_mobilenet_v3_large = segmentation.lraspp_mobilenet_v3_large(pretrained=True)
lraspp_mobilenet_v3_large = torch.nn.Sequential(
    *(list(lraspp_mobilenet_v3_large.children())[:-1])
)

if __name__ == "__main__":
    model = lraspp_mobilenet_v3_large
    print(model)
