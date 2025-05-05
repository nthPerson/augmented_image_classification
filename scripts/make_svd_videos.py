#!/usr/bin/env python
"""
Generate <num_frames>-frame SVD videos for every Tiny-ImageNet‐10 image and
save them as MP4 clips + a CSV manifest.
"""

import csv, os, yaml, torch
from pathlib import Path
from torchvision import transforms, io
from tqdm.auto import tqdm
from diffusers import StableVideoDiffusionPipeline
from datasets.tiny_imagenet import TinyImageNet10

CFG_PATH = "configs/train3d_config.yaml"
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


#*##*#*#*#* Stuff that didn't work but I can't bring myself to delete it below #*#*#*#*#*#*#*#*#**#*#


# import csv, yaml, torch, torchvision.io as io
# from pathlib import Path
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from tqdm.auto import tqdm
# from diffusers import StableVideoDiffusionPipeline
# from datasets.tiny_imagenet import TinyImageNet10
#
# # ── NEW: turn off Flash-SDP kernels ──────────────────────
# torch.backends.cuda.enable_flash_sdp(False)
# torch.backends.cuda.enable_mem_efficient_sdp(False)
#
# # ---------- config ----------
# CFG_PATH   = "configs/train3d_config.yaml"
# OUT_ROOT   = Path("./svd_videos")
# BATCH      = 1                 # started with 4, ran out of GPU memory
# NUM_STEPS  = 12
# SAVE_FPS   = 6                 # human-viewable speed
# # -----------------------------
#
# cfg   = yaml.safe_load(open(CFG_PATH))
# wnids = [l.strip() for l in open(cfg["wnids_file"])]
# NFRM  = cfg["num_frames"]
# dtype = torch.float16
#
# OUT_ROOT.mkdir(exist_ok=True)
#
# # ---- Stable Video Diffusion pipeline --------------------
# pipe = (StableVideoDiffusionPipeline
#         .from_pretrained(cfg["diffusion_model"], torch_dtype=torch.float16)
#         .to("cuda"))
# pipe.enable_model_cpu_offload()        # weights on CPU, activations on GPU
# # pipe.enable_attention_slicing()
#
# pipe.to("cpu")
#
#
# # ------------ spatial & tensor transforms ---------------
# spatial_tf = transforms.Compose([transforms.Resize(224),
#                                  transforms.CenterCrop(224)])
# to_uint8   = transforms.Compose([transforms.ToTensor(),
#                                  lambda t: t.mul_(255)])
#
# # ------------ custom collate for PIL --------------------
# def collate_pil(batch):
#     imgs, lbls = zip(*batch)
#     return list(imgs), torch.tensor(lbls)
# # -----------------------------------------------------------
#
# # -------- Per-split processing -------------------------------------------
# #
#
# def process_split(split):
#     ds = TinyImageNet10(cfg["data_root"], wnids, split=split, transform=None)
#     loader = DataLoader(ds, batch_size=BATCH, shuffle=False,
#                         num_workers=4, pin_memory=True, collate_fn=collate_pil)
#
#     (OUT_ROOT / split).mkdir(exist_ok=True)
#     csv_path = OUT_ROOT / f"{split}.csv"
#
#     idx = 0
#     with csv_path.open("w", newline="") as fcsv:
#         writer = csv.writer(fcsv)
#         writer.writerow(["video_path", "label"])
#
#         for imgs, lbls in tqdm(loader, desc=split):
#             with torch.no_grad():
#                 outs = pipe(imgs,
#                             num_inference_steps=NUM_STEPS,
#                             num_frames=NFRM,
#                             decode_chunk_size=1).frames   # list[list[PIL]]
#
#             for lbl, frames in zip(lbls, outs):
#                 frames = frames[:NFRM]
#                 tensor = torch.stack(
#                     [to_uint8(spatial_tf(f)) for f in frames], 0
#                 ).byte().permute(0, 2, 3, 1)         # T,H,W,C
#
#                 vfile = Path(split) / f"{idx:05d}.mp4"
#                 io.write_video(str(OUT_ROOT / vfile),
#                                tensor.cpu(), fps=SAVE_FPS,
#                                video_codec="libx264",
#                                options={"crf": "18", "pix_fmt": "yuv420p"})
#                 writer.writerow([vfile.as_posix(), int(lbl)])
#                 idx += 1
#
# # -------------------------------------------------------------------------
# process_split("train")
# process_split("val")
# print(f"***************************COMPLETE*************************************"
#       f"\n     All videos saved to {OUT_ROOT.resolve()}")





# def process_split(split):
#     ds = TinyImageNet10(cfg["data_root"], wnids, split=split, transform=None)
#     loader = DataLoader(ds, batch_size=BATCH,shuffle=False, num_workers=4, pin_memory=True,
#         collate_fn=collate_pil,
#     )
#
#     (OUT_ROOT / split).mkdir(exist_ok=True)
#     csv_path = OUT_ROOT / f"{split}.csv"
#
#     running_idx = 0
#     with csv_path.open("w", newline="") as fcsv:
#         writer = csv.writer(fcsv)
#         writer.writerow(["video_path", "label"])
#
#         for imgs, lbls in tqdm(loader, desc=split):
#             with torch.no_grad():
#                 outs = pipe(
#                     imgs,
#                     num_inference_steps=NUM_STEPS,
#                     num_frames=NFRM,
#                     decode_chunk_size=1,
#                 ).frames  # list[list[PIL]]
#
#             for lbl, frames in zip(lbls, outs):
#                 frames = frames[:NFRM]
#                 tensor = torch.stack(
#                     [to_uint8(spatial_tf(f)) for f in frames], 0
#                 ).byte().permute(0, 2, 3, 1)  # T,H,W,C
#
#                 vfile = Path(split) / f"{running_idx:05d}.mp4"
#                 io.write_video(
#                     str(OUT_ROOT / vfile),
#                     tensor.cpu(),
#                     fps=SAVE_FPS,
#                     video_codec="libx264",
#                     options={"crf": "18", "pix_fmt": "yuv420p"},
#                 )
#                 writer.writerow([vfile.as_posix(), int(lbl)])
#                 running_idx += 1




# CFG_PATH = "configs/train3d_config.yaml"
# OUT_DIR = Path("./svd_videos")  # each split gets its own sub-directory
# OUT_DIR.mkdir(exist_ok=True)
#
# cfg = yaml.safe_load(open(CFG_PATH))
# wnids = [line.strip() for line in open(cfg["wnids_file"])]
# nfrm = cfg["num_frames"]
# dtype = torch.float16
#
# # Build pipeline once
# pipe = StableVideoDiffusionPipeline.from_pretrained(cfg["diffusion_model"], torch_dtype=dtype).to("cuda")
# pipe.enable_model_cpu_offload()
#
# # Torchvision transform shared by all frames
# tf = transforms.Compose([
#     transforms.Resize(224),
#     transforms.CenterCrop(224),
# ])
#
# # Helper to generate and save one sample
# def process_split(split):
#     ds = TinyImageNet10(cfg["data_root"], wnids, split=split, transform=None)
#     sub = OUT_DIR / split
#     sub.mkdir(exist_ok=True)
#
#     csv_path = OUT_DIR / f"{split}.csv"
#     with csv_path.open("w", newline="") as fcsv:
#         writer = csv.writer(fcsv)
#         writer.writerow(["video_path", "label"])  # csv header
#
#         for idx, (img, lbl) in tqdm(enumerate(ds), total=len(ds), desc=f"{split}"):
#             with torch.no_grad():
#                 vid = pipe(img, num_inference_steps=20, num_frames=nfrm, decode_chunk_size=1).frames[0]
#
#             # vid = list[PIL]  -> apply spatial tf and stack to uint8 tensor
#             frames = [transforms.ToTensor()(tf(fr))*255 for fr in vid]      # C, H, W, 0-255
#             tensor = torch.stack(frames, dim=0).byte().permute(0, 2, 3, 1)  # T, W, H, C
#
#             vfile = sub / f"{idx:05d}.mp4"
#             io.write_video(str(vfile), tensor.cpu(), fps=24, video_codec="libx264", options={"crf":"18", "pix_fmt":"yuv420p"})
#
#             writer.writerow([vfile.name, lbl])
#
# process_split("train")
# process_split("val")
# print(f"All done! Videos saved to {OUT_DIR.resolve()}")


















