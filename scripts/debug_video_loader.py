import yaml
from torch.utils.data import DataLoader
from torchvision import transforms
from Inflated_3D_VGG.video_dataset import VideoClassificationDataset


# Load config
cfg = yaml.safe_load(open("configs/train3d_video.yaml"))

# Define the same frame-wise transforms
frame_tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.Normalize(cfg["mean"], cfg["std"]),
])

# Instantiate dataset and dataloader
ds = VideoClassificationDataset(
    root_dir=cfg["data_root"] + "/train",
    csv_file=cfg["train_csv"],
    transform=frame_tf
)
loader = DataLoader(ds, batch_size=1, shuffle=False)

# Fetch one sample and print its shape
for video, label in loader:
    print("Video tensor shape:", video.shape)  # [B, C, T, H, W]
    print("Label:", label.item())
    break