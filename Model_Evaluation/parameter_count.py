from Baseline_VGG.vgg_baseline import get_vgg11_baseline
from Inflated_3D_VGG.vgg_inflated import VGG11_3D

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

vgg2d = get_vgg11_baseline(num_classes=10, pretrained=False)
print(f"2D model parameters: {count_parameters(vgg2d):,}")

vgg3d = VGG11_3D(vgg2d, time_dim=3)
print(f"3D model parameters: {count_parameters(vgg3d):,}")