from torchvision.models import detection

fasterrcnn_resnet50_fpn = detection.fasterrcnn_resnet50_fpn(pretrained=True)
fasterrcnn_mobilenet_v3_large_fpn = detection.fasterrcnn_mobilenet_v3_large_fpn(
    pretrained=True
)
fasterrcnn_mobilenet_v3_large_320_fpn = detection.fasterrcnn_mobilenet_v3_large_320_fpn(
    pretrained=True
)
fcos_resnet50_fpn = detection.fcos_resnet50_fpn(pretrained=True)
retinanet_resnet50_fpn = detection.retinanet_resnet50_fpn(pretrained=True)
ssd300_vgg16 = detection.ssd300_vgg16(pretrained=True)
ssdlite320_mobilenet_v3_large = detection.ssdlite320_mobilenet_v3_large(pretrained=True)
maskrcnn_resnet50_fpn = detection.maskrcnn_resnet50_fpn(pretrained=True)
keypointrcnn_resnet50_fpn = detection.keypointrcnn_resnet50_fpn(pretrained=True)

if __name__ == "__main__":
    model = fasterrcnn_resnet50_fpn
    print(model)
