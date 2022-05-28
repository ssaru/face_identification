import torch
from torchvision import models


def resnet18() -> torch.nn.Module:
    model = models.resnet18(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def alexnet() -> torch.nn.Module:
    model = models.alexnet(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def squeezenet() -> torch.nn.Module:
    model = models.squeezenet1_0(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def vgg16() -> torch.nn.Module:
    model = models.vgg16(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def densenet() -> torch.nn.Module:
    model = models.densenet161(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def inception_v3() -> torch.nn.Module:
    model = models.inception_v3(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-3]))


def googlenet() -> torch.nn.Module:
    model = models.googlenet(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-3]))


def shufflenet() -> torch.nn.Module:
    model = models.shufflenet_v2_x1_0(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def mobilenet_v2() -> torch.nn.Module:
    model = models.mobilenet_v2(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def mobilenet_v3_large() -> torch.nn.Module:
    model = models.mobilenet_v3_large(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def mobilenet_v3_small() -> torch.nn.Module:
    model = models.mobilenet_v3_small(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def resnext50_32x4d() -> torch.nn.Module:
    model = models.resnext50_32x4d(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def wide_resnet50_2() -> torch.nn.Module:
    model = models.wide_resnet50_2(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def mnasnet() -> torch.nn.Module:
    model = models.mnasnet1_0(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def efficientnet_b7() -> torch.nn.Module:
    model = models.efficientnet_b7(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_y_400mf() -> torch.nn.Module:
    model = models.regnet_y_400mf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_y_800mf() -> torch.nn.Module:
    model = models.regnet_y_800mf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_y_1_6gf() -> torch.nn.Module:
    model = models.regnet_y_1_6gf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_y_3_2gf() -> torch.nn.Module:
    model = models.regnet_y_3_2gf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_y_8gf() -> torch.nn.Module:
    model = models.regnet_y_8gf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_y_16gf() -> torch.nn.Module:
    model = models.regnet_y_16gf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_y_32gf() -> torch.nn.Module:
    model = models.regnet_y_32gf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_x_400mf() -> torch.nn.Module:
    model = models.regnet_x_400mf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_x_800mf() -> torch.nn.Module:
    model = models.regnet_x_800mf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_x_1_6gf() -> torch.nn.Module:
    model = models.regnet_x_1_6gf(pretrained=True)
    return model  # torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_x_3_2gf() -> torch.nn.Module:
    model = models.regnet_x_3_2gf(pretrained=True)
    return model  # torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_x_8gf() -> torch.nn.Module:
    model = models.regnet_x_8gf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_x_16gf() -> torch.nn.Module:
    model = models.regnet_x_16gf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def regnet_x_32gf() -> torch.nn.Module:
    model = models.regnet_x_32gf(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def vit_b_16() -> torch.nn.Module:
    model = models.vit_b_16(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def vit_b_32() -> torch.nn.Module:
    model = models.vit_b_32(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def vit_l_16() -> torch.nn.Module:
    model = models.vit_l_16(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def vit_l_32() -> torch.nn.Module:
    model = models.vit_l_32(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-1]))


def convnext_tiny() -> torch.nn.Module:
    model = models.convnext_tiny(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def convnext_small() -> torch.nn.Module:
    model = models.convnext_small(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def convnext_base() -> torch.nn.Module:
    model = models.convnext_base(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


def convnext_large() -> torch.nn.Module:
    model = models.convnext_large(pretrained=True)
    return torch.nn.Sequential(*(list(model.children())[:-2]))


if __name__ == "__main__":
    m = convnext_large()
    print(m)
