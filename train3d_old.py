import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from diffusers import StableVideoDiffusionPipeline

from datasets.tiny_imagenet import TinyImageNet10
from datasets.tiny_imagenet_aug import AugmentedTinyImageNet10
from models.vgg_baseline import get_vgg11_baseline
from models.vgg_inflated import VGG11_3D
# from utils import save_checkpoint, average_meter, accuracy
from utils.utils import save_checkpoint, average_meter, accuracy

import torch.multiprocessing as mp

def main(config_path="configs/train3d_config.yaml"):
    # Load config
    cfg = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Read synset IDs
    wnids = [line.strip() for line in open(cfg["wnids_file"])]

    # Frame transforms
    frame_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    # Base datasets
    base_train = TinyImageNet10(cfg["data_root"], wnids, split="train", transform=None)
    base_val = TinyImageNet10(cfg["data_root"], wnids, split="val", transform=None)

    # Diffusion pipeline
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        cfg["diffusion_model"], torch_dtype=torch.float16
    ).to(device)

    # To save GPU memory: offload weights to CPU, move layers to GPU only
    pipe.enable_model_cpu_offload()
    pipe.to(dtype=torch.float16)  # make sure any new tensors default to half-precision

    # Augmented datasets and loaders
    train_ds = AugmentedTinyImageNet10(base_train, pipe,
                                       num_frames=cfg["num_frames"],
                                       transform=frame_tf)
    val_ds   = AugmentedTinyImageNet10(base_val,   pipe,
                                       num_frames=cfg["num_frames"],
                                       transform=frame_tf)
    train_loader = DataLoader(train_ds,
                              batch_size=cfg["batch_size"],
                              shuffle=True,
                              num_workers=cfg["workers"],
                              pin_memory=True)
    val_loader = DataLoader(val_ds,
                            batch_size=cfg["batch_size"],
                            shuffle=False,
                            num_workers=cfg["workers"],
                            pin_memory=True)

    # Models
    vgg2d = get_vgg11_baseline(num_classes=len(wnids), pretrained=True)
    model3d = VGG11_3D(vgg2d, time_dim=cfg["num_frames"]).to(device)

    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model3d.parameters(),
                                lr=cfg["lr"],
                                momentum=cfg["momentum"],
                                weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer,
                                                step_size=cfg["lr_step"],
                                                gamma=cfg["lr_gamma"])

    best_acc = 0.0
    for epoch in range(cfg["epochs"]):
        # Training
        model3d.train()
        train_loss = average_meter(); train_acc = average_meter()
        for vids, labels in train_loader:
            vids, labels = vids.to(device), labels.to(device)
            preds = model3d(vids)
            loss = criterion(preds, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, _ = accuracy(preds, labels, topk=(1,5))
            train_loss.update(loss.item(), vids.size(0))
            train_acc.update(acc1.item(), vids.size(0))

        print(f"Epoch {epoch} TRAIN  Loss: {train_loss.avg:4f}  Acc@1: {train_acc.avg:.2f}%")

        # Validation
        model3d.eval()
        val_loss = average_meter(); val_acc = average_meter()
        with torch.no_grad():
            for vids, labels in val_loader:
                vids, labels = vids.to(device), labels.to(device)
                preds = model3d(vids)
                loss = criterion(preds, labels)
                acc1, _ = accuracy(preds, labels, topk=(1,5))
                val_loss.update(loss.item(), vids.size(0))
                val_acc.update(acc1.item(), vids.size(0))

        print(f"Epoch {epoch} VALID  Loss {val_loss.avg:.4f}  Acc@1: {val_acc.avg:.2f}%")

        # Checkpoint
        is_best = val_acc.avg > best_acc
        best_acc = max(best_acc, val_acc.avg)
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model3d.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best, cfg["output_dir"])

        scheduler.step()

if __name__ == "__main__":
    # bugfix: 'spawn' start method
    mp.set_start_method("spawn", force=True)

    # Train
    main()


