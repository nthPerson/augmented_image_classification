import torch
import torch.nn as nn


def inflate_conv2d_to_conv3d(conv2d, time_dim=3):
    """
    Inflate a Conv2d layer into Conv3d by adding a time kernel and copying weights to central slice.
    """
    conv3d = nn.Conv3d(
        in_channels = conv2d.in_channels,
        out_channels = conv2d.out_channels,
        kernel_size = (time_dim,) + conv2d.kernel_size,
        stride = (1,) + conv2d.stride,
        padding = (0,) + conv2d.padding,
        bias = (conv2d.bias is not None)
    )
    mid = time_dim // 2
    with torch.no_grad():
        w2d = conv2d.weight.data
        w3d = torch.zeros_like(conv3d.weight.data)
        w3d[:, :, mid, :, :] = w2d
        conv3d.weight.data.copy_(w3d)
        if conv2d.bias is not None:
            conv3d.bias.data.copy_(conv2d.bias.data)
    return conv3d

class VGG11_3D(nn.Module):
    def __init__(self, baseline_vgg2d, time_dim=3):
        """
        baseline_vgg2d: a nn.Module VGG11 pretrained on 2D
        time_dim: number of frames
        """
        super().__init__()
        layers3d = []
        for layer in baseline_vgg2d.features:
            if isinstance(layer, nn.Conv2d):  # Inflate Conv2d to Conv3d
                layers3d.append(inflate_conv2d_to_conv3d(layer, time_dim))
            elif isinstance(layer, nn.BatchNorm2d): # Inflate BatchNorm2d layers to BatchNorm3d
                bn3d = nn.BatchNorm3d(
                    num_features=layer.num_features,
                    eps=layer.eps,
                    momentum=layer.momentum,
                    affine=layer.affine
                )
                bn3d.weight.data.copy_(layer.weight.data)
                bn3d.bias.data.copy_(layer.bias.data)
                layers3d.append(bn3d)
            elif isinstance(layer, nn.MaxPool2d):  # Inflate MaxPool2d layers to MaxPool3d
                layers3d.append(nn.MaxPool3d(
                    kernel_size=(1,) + layer.kernel_size,
                    stride=(1,) + layer.stride,
                    padding=(0,) + layer.padding
                ))
            else:
                # ReLU, Dropout, etc. remain unchanged
                layers3d.append(layer)
        self.features3d = nn.Sequential(*layers3d)
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 7, 7))
        self.classifier = baseline_vgg2d.classifier

    def forward(self, x):
        # x shape: [B, C, T, H, W]
        x = self.features3d(x)
        x = self.avgpool3d(x)      # [B, C, 1, 7, 7]
        x = torch.flatten(x, 1)    # [B, C*1*7*7]
        return self.classifier(x)

