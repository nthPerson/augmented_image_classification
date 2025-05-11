#!/usr/bin/env python
"""
Generate <num_frames>-frame SVD videos for every Tiny-ImageNet‐10 image and
save them as MP4 clips + a CSV manifest.
"""

import csv, yaml, torch
from pathlib import Path
from torchvision import transforms, io
from tqdm.auto import tqdm
from diffusers import StableVideoDiffusionPipeline
from Baseline_VGG.tiny_imagenet import TinyImageNet10

CFG_PATH = "configs/data_aug_config.yaml"
OUT_DIR  = Path("./svd_videos")           # each split gets its own sub-dir
OUT_DIR.mkdir(exist_ok=True)

cfg   = yaml.safe_load(open(CFG_PATH))
wnids = [l.strip() for l in open(cfg["wnids_file"])]
nfrm  = cfg["num_frames"]
dtype = torch.float16

# -------- build pipeline once --------------------------------------------------
pipe = (StableVideoDiffusionPipeline
        .from_pretrained(cfg["diffusion_model"], torch_dtype=dtype)
        .to("cuda"))
pipe.enable_model_cpu_offload()           # keep weights off GPU

# -------- torchvision transform shared by all frames --------------------------
tf = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
])                                         # leave ToTensor/Normalize to train time

# -------- helper to generate & save one sample --------------------------------
def process_split(split):
    ds   = TinyImageNet10(cfg["data_root"],
                          wnids, split=split, transform=None)
    sub  = OUT_DIR / split
    sub.mkdir(exist_ok=True)

    csv_path = OUT_DIR / f"{split}.csv"
    with csv_path.open("w", newline="") as fcsv:
        writer = csv.writer(fcsv)
        writer.writerow(["video_path", "label"])      # header

        for idx, (img, lbl) in tqdm(enumerate(ds),
                                    total=len(ds),
                                    desc=f"{split}"):
            with torch.no_grad():
                vid = pipe(img,
                           num_inference_steps=12,
                           num_frames=nfrm,
                           decode_chunk_size=1).frames[0]

            # vid = list[PIL]  → apply spatial tf & stack to uint8 Tensor
            frames = [transforms.ToTensor()(tf(fr))*255 for fr in vid]   # C,H,W,0-255
            tensor = torch.stack(frames, dim=0).byte().permute(0,2,3,1)  # T,H,W,C

            vfile  = sub / f"{idx:05d}.mp4"
            io.write_video(str(vfile),
                           tensor.cpu(),           # uint8
                           fps=6,
                           video_codec="libx264",
                           options={"crf":"18", "pix_fmt":"yuv420p"})   # visually lossless

            writer.writerow([vfile.name, lbl])

# -------------------------------------------------------------------------


process_split("train")
process_split("val")
print(f"***************************COMPLETE*************************************"
      f"\n     All videos saved to {OUT_DIR.resolve()}")

