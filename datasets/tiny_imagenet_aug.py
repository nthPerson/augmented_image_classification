import torch
from torch.utils.data import Dataset


class AugmentedTinyImageNet10(Dataset):
    """
    Wrap a TinyImageNet10 dataset to generate T-frame videos per sample via a Stable Diffusion pipeline.
    Returns: Tensor video [C, T, H, W], int label
    """
    def __init__(self, base_dataset, pipeline, num_frames=3, transform=None):
        """
        base_dataset: instance of TinyImageNet10 (returns PIL image, label)
        pipeline: a StableVideoDiffusionPipeline from diffusers
        num_frames: total frames (including original) to return
        transform: torchvision transforms to apply to each frame
        """
        self.base = base_dataset
        self.pipe = pipeline
        self.num_frames = num_frames
        self.transform = transform

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        # Get original image and label
        img, label = self.base[idx]  # PIL.Image
        # Generate video frames (num_frames total)
        with torch.no_grad():
            out = self.pipe(img, num_inference_steps=20, num_frames=self.num_frames)
        # frames = out.frames  # list of PIL.Image length=num_frames
        frames = out.frames[0]  # list of PIL.Image length=num_frames (bugfix: select first element instead of getting 1-element list)
        # Apply transform to each and stack
        tensors = [self.transform(frame) for frame in frames]
        video = torch.stack(tensors, dim=1)  # Shape [C, T, H, W]
        return video, label














