# Multimodal Oral Cancer Cell Classification

Comparison of deep learning models that combine **brightfield (BF)** and **fluorescence (FL)** microscopy images to classify cells as cancer or not cancer. This was my work for a course challenge in *Advanced Deep Learning for Image Processing* at Uppsala University.

> The dataset is not included and is not mine to share. It comes from the course challenge (Kaggle: `multimodal-cancer-classification-challenge-2026`). The notebooks are exploratory and written as experiments, not as a finished package.

## Goal
Test whether using both imaging modalities helps, and which way of combining them works best.

## Data (not included)
- Paired BF and FL images of cells, with a binary label (cancer / not cancer).
- About 114,000 images from 12 patients.
- Because all images from one patient look alike, splitting by image would leak information.

## Evaluation setup
- **Patient-level cross-validation** with `GroupKFold` (3 to 5 folds depending on the notebook), so no patient appears in both training and validation.
- With only 12 patients, each fold holds 2 to 4 patients, so results vary a lot between folds.
- Metrics: AUROC (main), accuracy, and class-wise precision/recall.

## Models
| Notebook | Model | Fusion |
|---|---|---|
| 01 | ResNet50 | Baseline |
| 02 | Two ResNet50 backbones | Late (feature concatenation) |
| 03 | EfficientNet-B0 (one per modality) | Late |
| 04 | CAFNet (ResNet backbones + cross-attention) | Cross-attention |
| 05 | ConvNeXt V2 tiny | 6-channel input and dual-backbone fusion |
| 06 | EfficientNet-B0, ResNet18, ViT-tiny, Swin | Early and late fusion |

## Training details
- PyTorch and `timm` with pretrained backbones, AdamW, `ReduceLROnPlateau`, mixed precision.
- Augmentation with Albumentations, applied **identically to both modalities** (flips, rotations, brightness, hue).
- Mixup and label smoothing in some runs, class-weighted loss in some runs.
- Image size 128 or 224 pixels.

## Observations
- Final held-out score 0.8057, rank 8
- Training AUROC often reaches about 0.99 while validation accuracy stays much lower, which shows strong overfitting to the training patients.
- Validation AUROC is higher than accuracy in several runs, which suggests the decision threshold matters when the class balance differs between folds.
- With so few patients, differences between models are small compared with the variation between folds.

## Limitations
- Only 12 patients, so the results have high variance.
- Hyperparameters were not tuned systematically, and most results come from single runs.
- The validation folds are small, and the class balance differs between them.

## How to run
1. Get access to the challenge data and place it as described in the first cell.
2. `pip install -r requirements.txt`
3. Run a notebook from the `notebooks/` folder.

## Possible next steps
- Repeated cross-validation with several seeds, and confidence intervals.
- Brightfield-only and fluorescence-only baselines, to measure what each modality adds.
- Choose the decision threshold on the validation folds, and report precision and recall alongside AUROC.
- Stronger regularisation or pretrained pathology models to reduce overfitting.

## Author
Md Fahim, MSc Image Analysis and Machine Learning, Uppsala University.
