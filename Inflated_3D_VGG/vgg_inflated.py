import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair

def inflate_conv2d_to_conv3d(conv2d, time_dim=3):
    """
    Inflate a Conv2d layer into Conv3d by adding a time kernel and copying weights to central slice.
    Preserves the temporal dimension by padding equally on both sides.
    """
    # compute padding for time dimension to preserve time length
    time_pad = time_dim // 2

    conv3d = nn.Conv3d(
        in_channels  = conv2d.in_channels,
        out_channels = conv2d.out_channels,
        kernel_size  = (time_dim,) + conv2d.kernel_size,
        stride       = (1,)         + conv2d.stride,
        padding      = (time_pad,)  + conv2d.padding,
        bias         = (conv2d.bias is not None)
    )
    # initialize weights: copy 2D weights into the central time slice
    mid = time_dim // 2
    with torch.no_grad():
        w2d = conv2d.weight.data  # shape [out, in, kH, kW]
        w3d = torch.zeros_like(conv3d.weight.data)  # [out, in, kT, kH, kW]
        w3d[:, :, mid, :, :] = w2d
        conv3d.weight.data.copy_(w3d)
        if conv2d.bias is not None:
            conv3d.bias.data.copy_(conv2d.bias.data)
    return conv3d

class VGG11_3D(nn.Module):
    def __init__(self, baseline_vgg2d, time_dim=3):
        """
        baseline_vgg2d: a nn.Module VGG11 pretrained on 2D
        time_dim: number of frames (temporal depth)
        """
        super().__init__()
        layers3d = []
        for layer in baseline_vgg2d.features:
            print(f'\nInflating layer: {layer}')
            print(f'-----Layer type: {type(layer)}')
            if isinstance(layer, nn.Conv2d):
                layers3d.append(inflate_conv2d_to_conv3d(layer, time_dim))
                print(f'----------Converted from Conv2d to Conv3d')
            elif isinstance(layer, nn.BatchNorm2d):
                # convert to BatchNorm3d
                bn3d = nn.BatchNorm3d(
                    num_features=layer.num_features,
                    eps=layer.eps,
                    momentum=layer.momentum,
                    affine=layer.affine
                )
                # copy weights and bias
                bn3d.weight.data.copy_(layer.weight.data)
                bn3d.bias.data.copy_(layer.bias.data)
                layers3d.append(bn3d)
                print(f'----------Converted from BatchNorm2d to BatchNorm3d')
            elif isinstance(layer, nn.MaxPool2d):
                # inflate MaxPool2d to MaxPool3d: no time pooling
                layers3d.append(nn.MaxPool3d(
                    kernel_size=(1,) + _pair(layer.kernel_size),
                    stride=(1,) + _pair(layer.stride),
                    padding=(0,) + _pair(layer.padding),
                ))
            else:
                # ReLU, Dropout remain unchanged; works since they ignore extra time dim
                layers3d.append(layer)
                print(f'----------Layer unchanged: {type(layer)}')

            print(f'-----Layer 3D type: {type(layers3d[-1])}')
        self.features3d = nn.Sequential(*layers3d)

        # pool to (T, H, W) = (time_dim, 7, 7) -> but we want to preserve time until the end
        # then flatten T*7*7, but original avgpool used (1,7,7)
        self.avgpool3d = nn.AdaptiveAvgPool3d((1, 7, 7))

        # reuse classifier as-is
        self.classifier = baseline_vgg2d.classifier



    def forward(self, x):
        # x shape: [B, C, T, H, W]
        x = self.features3d(x)
        # debug: ensure temporal dim remains = original T
        # print(f"After features3d: {x.shape}")
        x = self.avgpool3d(x)           # [B, C, 1, 7, 7]
        x = torch.flatten(x, 1)         # [B, C*1*7*7]
        return self.classifier(x)