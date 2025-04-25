
import torch, platform, os
print("Torch version:", torch.__version__)
print("CUDA available? ", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    a = torch.randn(4096, 4096, device='cuda')
    b = torch.mm(a, a)
    torch.cuda.synchronize()
    print("Matrix mul OK, result sum:", b.sum().item())

# import torch, sys; print(torch.__version__, "CUDA OK?", torch.cuda.is_available())
# if torch.cuda.is_available():
#     print("Device:", torch.cuda.get_device_name(0))
