import torch.nn as nn
from torchvision import models


def get_vgg11_baseline(num_classes=10, pretrained=True):
    model = models.vgg11_bn(pretrained=pretrained)

    # Swap the final classifier layer for our 10-class implementation
    model.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)

    return model
