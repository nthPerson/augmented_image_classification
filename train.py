import os
import yaml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets.tiny_imagenet import TinyImageNet10
from models.vgg_baseline import get_vgg11_baseline
from utils import save_checkpoint, average_meter, accuracy

def main(config_path="configs/train_config.yaml"):
    # --- 1. Load config ---
    cfg = yaml.safe_load(open(config_path))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- 2. Data transforms & loaders ---
    wnids = [line.strip() for line in open(cfg["wnids_file"])]
    train_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])
    val_tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(cfg["mean"], cfg["std"]),
    ])

    train_ds = TinyImageNet10(cfg["data_root"], wnids, split="train", transform=train_tf)
    val_ds = TinyImageNet10(cfg["data_root"], wnids, split="val", transform=val_tf)

    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True, num_workers=cfg["workers"])
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False, num_workers=cfg["workers"])

    # --- 3. Model, loss function, optimizer, scheduler ---
    model = get_vgg11_baseline(num_classes=len(wnids), pretrained=True)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=cfg["lr"], momentum=cfg["momentum"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=cfg["lr_step"], gamma=cfg["lr_gamma"])

    best_acc = 0.0
    for epoch in range(cfg["epochs"]):
        # Train
        model.train()
        losses = average_meter()
        top1 = average_meter()
        for imgs, targets in train_loader:
            imgs, targets = imgs.to(device), targets.to(device)
            preds = model(imgs)
            loss = criterion(preds, targets)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc1, _ = accuracy(preds, targets, topk=(1, 5))
            losses.update(loss.item(), imgs.size(0))
            top1.update(acc1.item, imgs.size(0))

        print(f"Epoch {epoch} TRAIN  Loss: {losses.avg:.4f}  Top1: {top1.avg:.2f}%")

        # Validate
        model.eval()
        val_losses = average_meter()
        val_top1 = average_meter()
        with torch.no_grad():
            for imgs, targets in val_loader:
                imgs, targets = imgs.to(device), targets.to(device)
                preds = model(imgs)
                loss = criterion(preds, targets)

                acc1, _ = accuracy(preds, targets, topk=(1, 5))
                val_losses.update(loss.item(), imgs.size(0))
                val_top1.update(acc1.item(), imgs.size(0))

        print(f"Epoch {epoch} VALID  Loss: {val_losses.avg:.4f}  Top1: {val_top1.avg:.2f}%")

        # Checkpoint
        is_best = val_top1.avg > best_acc
        best_acc = max(best_acc, val_top1.avg)
        save_checkpoint({
            "epoch": epoch + 1,
            "state_dict": model.state_dict(),
            "best_acc": best_acc,
            "optimizer": optimizer.state_dict(),
        }, is_best, cfg["output_dir"])

        scheduler.step()

if __name__ == "__main__":
    main()
