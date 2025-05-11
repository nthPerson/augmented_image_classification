# Generative Video Augmentation for Image Classification
## CS 559 Computer Vision Group Project

### Group Members: Robert Ashe, Afnan Algharby, Bieu To, Adam Lizerbram, Thiago Henriques


## Project Overview
This repository contains the code and data used in the project described in Generative_Video_Augmentation_for_Image_Classification.pdf. In brief, we show that by using a generative model to hallucinate a tiny “pseudo‐video” (just two extra frames) from each static image, and then training an inflated 3D VGG-11 network on these 3-frame clips, you can recover motion cues and boost classification accuracy by over 20 percentage points on a 10-class Tiny ImageNet subset—at the cost of only ~0.14 ms extra inference time per sample.

## Dataset Availability
The dataset that we generated for this project is available on Kaggle. You can download it from the following link: https://www.kaggle.com/datasets/afnanalgarby/svd-generated-video-dataset/data

## Project Structure
├── Baseline_VGG/             # Data loaders, model definition, training script & config for the 2D VGG-11 baseline
<p>├── Data_Augmentation_SVD/ # Script and config used to generate 3-frame video dataset via Stable Video Diffusion
<p>├── Data_Sourcing/         # Code to extract and filter the 10-class Tiny ImageNet subset
<p>├── Inflated_3D_VGG/       # Data loaders, model definition, training script & config for the 3D‐inflated VGG-11
<p>├── Model_Evaluation/      # Inference scripts, metric calculators, FLOPs/parameter counters
<p>├── scripts/               # Throw-away/debug scripts (e.g. video loader tests)
<p>├── svd_videos/            # The generated 3-frame “video” dataset (train/val + CSV indices)
<p>├── utils/                 # Utility functions used across the repo
<p>├── Generative_Video_Augmentation_for_Image_Classification.pdf # The final report
<p>├── README.md              # You are here
<p>└── requirements.txt # Python dependencies

### `Baseline_VGG/`
- **`tiny_imagenet.py`** – Dataset definition for the 10-class Tiny ImageNet subset.
- **`vgg_baseline.py`** – VGG-11 model wrapper, head replacement for 10 classes.
- **`train.py`** & - Training script for the 2D baseline model.
- **`train_config.yaml`** – Hyperparameter YAML config for 2D baseline.

### `Data_Augmentation_SVD/`
- **`make_svd_videos.py`** – Data augmentation entry point: loads each 224×224 image, invokes SVD to generate two extra frames, saves 3-frame clips.
- **`tiny_imagenet_aug.py`** – Dataset definition for the SVD data augmentation process.
- **`train3d_config.yaml`** – Data augmentation YAML config: number of frames to generate, model and data IDs.

### `Data_Sourcing/`
- **`filter.py`** – Filters ImageNet synsets to our 10 mammal classes, copies train images.
- **`val.py`** – Builds the 500-image validation set from the original Tiny ImageNet val folder.
- **`README-data_sourcing.txt`** – Detailed information about data curation process.

### `Inflated_3D_VGG/`
- **`video_dataset.py`** – PyTorch dataset for loading 3×224×224 clips + labels.
- **`vgg_inflated.py`** – Converts a pretrained 2D VGG-11 to 3D VGG-11 (Conv3D/BatchNorm3D/MaxPool3D).
- **`train3d.py`** - Training script for the inflated 3D model.
- **`train3d_video.yaml`** – Hyperparameter YAML config for the inflated 3D model.

### `Model_Evaluation/`
- **`evaluate.py`** Runs inference on baseline vs. inflated model; computes Top-k accuracy and other metrics.
- **`evaluate.yaml`** – YAML config for evaluation script: model IDs, output CSVs.
- **`prepare_evaluation_data.py`** – Transforms evaluation data for visualization in Tableau.
- **`parameter_count.py`** & **`flops_count.py`** – Scripts to report model size and computational cost.
- **`evaluation_results/`** – Stored confusion matrices, per-class metrics, and other evaluation data.

### `scripts/`
- **`debug_video_loader.py`** – Quick test to verify 3D `DataLoader` & transforms.

### `svd_videos/`
- **`train/`** & **`val/`** – The actual 3-frame videos.
- **`train.csv`**, **`val.csv`** – Filename-to=label indices for augmented dataset.

### `utils/`
- **`utils.py`** – Common helpers (seed setting, progress bars, metric aggregators).

- **`requirements.txt`** – Exact `pip` versions tested.
