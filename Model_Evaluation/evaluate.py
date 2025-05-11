import os
import time
import yaml
import json
import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    accuracy_score, top_k_accuracy_score,
    precision_recall_fscore_support,
    confusion_matrix
)

from Inflated_3D_VGG.video_dataset import VideoClassificationDataset
from Baseline_VGG.vgg_baseline import get_vgg11_baseline
from Inflated_3D_VGG.vgg_inflated import VGG11_3D

def load_model(cfg, device):
    """
    cfg: dict with keys:
      - type: 'baseline' or 'inflated'
      - checkpoint: path to .pth.tar
      - num_classes
      - data.time_dim
    """
    if cfg['type'] == 'baseline':
        model = get_vgg11_baseline(num_classes=cfg['num_classes'], pretrained=False)
    else:
        base = get_vgg11_baseline(num_classes=cfg['num_classes'], pretrained=False)
        model = VGG11_3D(base, time_dim=cfg['data']['time_dim'])
    ckpt = torch.load(cfg['checkpoint'], map_location=device)
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    model.eval()
    print(f'Loaded {cfg["type"]} model from {cfg["checkpoint"]}')
    return model

def evaluate(model, loader, device, model_type, time_dim, num_classes):
    """
    model_type: 'baseline' or 'inflated'
    time_dim: number of frames expected by 3D model (e.g. 3)

    Calculates accuracy, top-k accuracy, precision, recall, f1-score, and confusion matrix.

    Uses the first frame for the 2D model and the full video for the 3D model from the svd_videos dataset.
    """
    print(f'\nStarting {model_type} model evaluation on {len(loader)} samples')
    y_true, y_pred, y_score = [], [], []
    inference_times = []

    # middle = time_dim // 2  # don't want to use this because I want to use the first frame for the 2D model
    with torch.no_grad():
        print(f'---- Starting inference...')
        for vids, labels in loader:
            vids = vids.to(device) # shape: [B, C, T, H, W] (B=batch size, C=channels, T=num_frames, H=height, W=width)
            labels = labels.to(device) # shape: [B]

            # branch for baseline vs inflated model
            if model_type == 'baseline':
                # pick the first frame from the video (this was the original input for the 2D model)
                imp = vids[:, :, 0, :, :] # shape: [B, C, H, W]
                # vids = vids[:, :, middle, :, :]  # shape: [B, C, H, W]
            else:
                # keep full video as input: [B, C, T, H, W]
                imp = vids

            start = time.time()
            logits = model(imp)
            end = time.time()

            # record per-sample inference time
            inference_times.extend([(end - start) / vids.size(0)] * vids.size(0))

            probs = torch.softmax(logits, dim=1).cpu().numpy() # shape: [B, num_classes]
            preds = np.argmax(probs, axis=1) # shape: [B]

            y_true.extend(labels.cpu().tolist())
            y_pred.extend(preds.tolist())
            y_score.extend(probs.tolist())

    # compute metrics
    print(f'---- Finished inference, computing metrics...')
    top1 = accuracy_score(y_true, y_pred)
    top3 = top_k_accuracy_score(y_true, np.array(y_score), k=3, labels=list(range(num_classes)))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=list(range(num_classes)), zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))

    performance_metrics = {
        'top1': top1,
        'top3': top3,
        'precision': prec.tolist(),
        'recall': rec.tolist(),
        'f1': f1.tolist(),
        'confusion_matrix': cm.tolist(),
        'inference_times': inference_times
    }
    print(f'---- Performance metrics: {performance_metrics}\n')
    return performance_metrics

def main(config_path='configs/evaluate.yaml'):
    cfg = yaml.safe_load(open(config_path))
    os.makedirs(cfg['output_dir'], exist_ok=True)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # prepare validation loader
    tf = transforms.Compose([
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
    ])

    val_ds = VideoClassificationDataset(
        root_dir=os.path.join(cfg['data']['root'], 'val'),
        csv_file=cfg['data']['val_csv'],
        transform=tf
    )

    val_loader = DataLoader(
        val_ds, batch_size=cfg['data']['batch_size'],
        shuffle=False, num_workers=cfg['data']['workers'],
        pin_memory=True
    )

    print(f'\n---- Generating results...')
    summary = {}
    for key in ['model2d', 'model3d']:
        print(f'\n---- Evaluating {key}...')
        model_cfg = cfg[key]
        model_cfg['num_classes'] = cfg['num_classes']
        model_cfg['data'] = cfg['data']
        model = load_model(model_cfg, device)

        metrics = evaluate(
            model, val_loader, device,
            model_type=model_cfg['type'],
            time_dim=model_cfg['data']['time_dim'],
            num_classes=cfg['num_classes']
        )

        # Save JSON
        out = cfg['output_dir']
        with open(f'{out}/{key}_metrics.json', 'w') as f:
            print(f'---- ---- Saving {key} metrics to {out}/{key}_metrics.json')
            json.dump(metrics, f, indent=2)

        # Save confusion matrix CSV
        pd.DataFrame(metrics['confusion_matrix']).to_csv(f'{out}/{key}_confusion.csv', index=False)
        print(f'---- ---- Saving {key} confusion matrix to {out}/{key}_confusion.csv')

        # Save per-class precision/recall/f1 report
        pd.DataFrame({
            'precision': metrics['precision'],
            'recall': metrics['recall'],
            'f1': metrics['f1']
        }).to_csv(f'{out}/{key}_precision_recall_f1_report.csv', index=False)
        print(f'---- ---- Saving {key} precision/recall/f1 report to {out}/{key}_precision_recall_f1_report.csv')

        # Save inference times list
        np.savetxt(f'{out}/{key}_inference_times.csv',
                   metrics['inference_times'], delimiter=',')
        print(f'---- ---- Saving {key} inference times to {out}/{key}_inference_times.csv')

        # Record summary metrics
        summary[key] = {
            'top1': metrics['top1'],
            'top3': metrics['top3'],
            'avg_inference_time': float(np.mean(metrics['inference_times']))
        }
        print(f'---- ---- Recording summary metrics for {key}')

    # Save summary metrics JSON
    with open(f'{cfg["output_dir"]}/summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f'---- Saving summary metrics for both models to {cfg["output_dir"]}/summary.json')

if __name__ == '__main__':
    main()
























# import os
# import time
# import yaml
# import json
# import torch
# import numpy as np
# from torch.utils.data import DataLoader
# from torchvision import transforms
# from datasets.video_dataset import VideoClassificationDataset
# from models.vgg_baseline import get_vgg11_baseline
# from models.vgg_inflated import VGG11_3D
# from sklearn.metrics import (
#     accuracy_score, top_k_accuracy_score,
#     precision_recall_fscore_support,
#     confusion_matrix
# )
# import pandas as pd
#
# def load_model(cfg, device):
#     if cfg['type'] == 'baseline':
#         model = get_vgg11_baseline(num_classes=cfg['num_classes'], pretrained=False)
#     else:
#         base = get_vgg11_baseline(num_classes=cfg['num_classes'], pretrained=False)
#         model = VGG11_3D(base, time_dim=cfg['data']['time_dim'])
#     checkpoint = torch.load(cfg['checkpoint'], map_location=device)
#     model.load_state_dict(checkpoint['state_dict'])
#
#     print(f'Loaded model from {cfg["checkpoint"]}')
#     return model.to(device).eval()
#
# def evaluate(model, loader, device, time_dim, num_classes):
#     print(f'Starting evaluation on {len(loader)} samples')
#     y_true, y_pred, y_score = [], [], []
#     inference_times = []
#     with torch.no_grad():
#         for vids, labels in loader:
#             vids, labels = vids.to(device), labels.to(device)
#             start = time.time()
#             logits = model(vids)
#             end = time.time()
#             inference_times.append((end - start) / vids.size(0))
#             probs = torch.softmax(logits, dim=1).cpu().numpy()
#             preds = np.argmax(probs, axis=1)
#             y_true.extend(labels.cpu().numpy())
#             y_pred.extend(preds.tolist())
#             y_score.extend(probs.tolist())
#     # Metrics
#     top1 = accuracy_score(y_true, y_pred)
#     top5 = top_k_accuracy_score(y_true, np.array(y_score), k=5, labels=list(range(num_classes)))
#     prec, rec, f1, _ = precision_recall_fscore_support(
#         y_true, y_pred, labels=list(range(num_classes)), zero_division=0
#     )
#     cm = confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))
#     performance_metrics = {
#         'top1': top1,
#         'top5': top5,
#         'precision': prec.tolist(),
#         'recall': rec.tolist(),
#         'f1': f1.tolist(),
#         'confusion_matrix': cm.tolist(),
#         'inference_times': inference_times
#     }
#     print(f'Performance metrics: {performance_metrics}')
#     return performance_metrics
#     # return {
#     #     'top1': top1,
#     #     'top5': top5,
#     #     'precision': prec.tolist(),
#     #     'recall': rec.tolist(),
#     #     'f1': f1.tolist(),
#     #     'confusion_matrix': cm.tolist(),
#     #     'inference_times': inference_times
#     # }
#
# def main(config_path='configs/evaluate.yaml'):
#     cfg = yaml.safe_load(open(config_path))
#     os.makedirs(cfg['output_dir'], exist_ok=True)
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     # Data loader
#     tf = transforms.Compose([
#         transforms.Resize(224),
#         transforms.CenterCrop(224),
#         transforms.Normalize(cfg['data']['mean'], cfg['data']['std'])
#     ])
#     val_ds = VideoClassificationDataset(
#         root_dir=os.path.join(cfg['data']['root'], 'val'),
#         csv_file=cfg['data']['val_csv'],
#         transform=tf
#     )
#     val_loader = DataLoader(
#         val_ds, batch_size=cfg['data']['batch_size'],
#         shuffle=False, num_workers=cfg['data']['workers'],
#         pin_memory=True
#     )
#
#     print(f'Generating results...')
#     results = {}
#     for key in ['model2d', 'model3d']:
#         print(f"Evaluating {key}...")
#         model_cfg = cfg[key]
#         model_cfg['num_classes'] = cfg['num_classes']
#         model_cfg['data'] = cfg['data']
#         model = load_model(model_cfg, device)
#         metrics = evaluate(model, val_loader, device,
#                            cfg['data']['time_dim'], cfg['num_classes'])
#         # Save JSON
#         out_path = os.path.join(cfg['output_dir'], f'{key}_metrics.json')
#         with open(out_path, 'w') as f:
#             json.dump(metrics, f, indent=2)
#         # Save confusion matrix CSV
#         cm_df = pd.DataFrame(metrics['confusion_matrix'])
#         cm_df.to_csv(os.path.join(cfg['output_dir'], f'{key}_confusion.csv'), index=False)
#         # Save per-class report
#         report_df = pd.DataFrame({
#             'precision': metrics['precision'],
#             'recall': metrics['recall'],
#             'f1': metrics['f1']
#         })
#         report_df.to_csv(os.path.join(cfg['output_dir'], f'{key}_report.csv'), index=False)
#         # Save inference times list
#         np.savetxt(os.path.join(cfg['output_dir'], f'{key}_inference_times.csv'),
#                    metrics['inference_times'], delimiter=',')
#         results[key] = metrics
#
#     # Optional: combine summary
#     summary = {
#         key: {'top1': results[key]['top1'], 'top5': results[key]['top5'],
#               'avg_inference_time': float(np.mean(results[key]['inference_times']))}
#         for key in results
#     }
#     with open(os.path.join(cfg['output_dir'], 'summary.json'), 'w') as f:
#         json.dump(summary, f, indent=2)

# if __name__ == '__main__':
#     main()












# Usage:
# python evaluate.py --config configs/evaluate.yaml