import os
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.io import read_video


class VideoClassificationDataset(Dataset):
    """
    Dataset for preprocessed 3D videos stored as .mp4 files,
    with filename->label mappings in a CSV.
    Returns: video tensor [C, T, H, W], int label
    """
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            root_dir (string): Directory containing video files.
            csv_file (string): Path to the CSV with columns "video_path,label"
            transform: torchvision transform to apply to each frame
        """
        self.root_dir = root_dir
        self.df = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        video_path = os.path.join(self.root_dir, row['video_path'])
        label = int(row['label'])

        # Load all frames from the .mp4 (assumes exactly `T` frames per video)
        video_frames, _, _ = read_video(video_path, pts_unit='sec')
        # video_frames: Tensor[T, H, W, C]

        # Convert to float and scale to [0,1]
        video_frames = video_frames.permute(0, 3, 1, 2).float() / 255.0  # (T, C, H, W)
        # video_frames: Tensor[T, C, H, W]

        # Apply transforms to each frame
        processed = []
        for frame in video_frames:
            if self.transform:
                frame = self.transform(frame)
            processed.append(frame)
        # Stack frames back into a single tensor [C, T, H, W]
        video = torch.stack(processed, dim=1)  # (C, T, H, W)

        return video, label


