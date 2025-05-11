import torch
from ptflops import get_model_complexity_info

from models.vgg_baseline import get_vgg11_baseline
from models.vgg_inflated import VGG11_3D

# For 2D model
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        get_vgg11_baseline(num_classes=10, pretrained=False),
        (3, 224, 224),
        as_strings=True, print_per_layer_stat=False)
    print(f'2D Model: MACs = {macs}, Params = {params}')

# For 3D model
model3d = VGG11_3D(get_vgg11_baseline(num_classes=10, pretrained=False), time_dim=3)
with torch.cuda.device(0):
    macs, params = get_model_complexity_info(
        model3d,
        (3, 3, 224, 224),  # (C, T, H, W)
        as_strings=True, print_per_layer_stat=False)
    print(f'3D Model: MACs = {macs}, Params = {params}')