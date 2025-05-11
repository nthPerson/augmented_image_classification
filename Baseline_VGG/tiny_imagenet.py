import os
from PIL import Image
from torch.utils.data import Dataset

class TinyImageNet10(Dataset):
    def __init__(self, root_dir, wnids, split="train", transform=None):
        self.transform = transform
        self.class_to_idx = {syn: i for i, syn in enumerate(wnids)}
        self.samples = []

        if split == "train":
            for syn in wnids:
                img_dir = os.path.join(root_dir, "train", syn, "images")
                for fn in os.listdir(img_dir):
                    if fn.lower().endswith(".jpeg"):
                        self.samples.append((os.path.join(img_dir, fn),
                                             self.class_to_idx[syn]))
        else:  # val
            ann_file = os.path.join(root_dir, "val", "val_annotations.txt")
            ann = {}
            with open(ann_file) as f:
                for line in f:
                    fn, syn, *_ = line.strip().split()
                    if syn in self.class_to_idx:
                        ann[fn] = self.class_to_idx[syn]
            img_dir = os.path.join(root_dir, "val", "images")
            for fn, lbl in ann.items():
                path = os.path.join(img_dir, fn)
                if os.path.exists(path):
                    self.samples.append((path, lbl))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


