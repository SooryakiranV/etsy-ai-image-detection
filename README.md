# AI-Generated Image Detection for Etsy Product Listings

## Overview

This project detects whether an Etsy product image is authentic or AI-generated using a multi-model ensemble.

## Models Used

- EfficientNet-B4
- Vision Transformer (ViT-B/16)
- ConvNeXt-Base
- Frequency-Domain Classifier

## Key Features

- 5-Fold Stratified Cross Validation
- Transfer Learning
- Mixup & CutMix
- Temperature Scaling
- Test-Time Augmentation
- Ensemble Weight Optimization
- Threshold Optimization

## Results

| Model | OOF F1 |
|---------|---------|
| EfficientNet-B4 | 0.7884 |
| ConvNeXt-Base | 0.8785 |
| ViT-B/16 | 0.8853 |
| Frequency Classifier | 0.9145 |
| Final Ensemble | 0.9270 |

## Architecture

(Insert architecture diagram image)

## Technologies

- Python
- PyTorch
- timm
- Albumentations
- Scikit-Learn
- OpenCV
