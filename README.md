# Which ViT to favor for Image Segmentation Model  

This code is adapted from the [📄 Paper](https://arxiv.org/abs/2503.19108) to use it as a common framework for comparing different variants of Vision Transformers for the task of image segmentation.

Please refer to the original in case you were mislead -> [Original Repo](https://github.com/tue-mps/eomt)



## Overview

Similar to the Benchmarking project at [Which Transformer to favor](https://github.com/tobna/WhatTransformerToFavor). This project aims to create a common framework that can compare variants of Vision Transformers for the task of image segmentation.

## Adaptations

- Decoupled Queries and Image Features processing to allow backbone flexibility: `models\eomt.py`
- One way Cross-attention between queries derived from images vs. key/values from image features: `models\cross_attention.py`
- Change Learning scheduler to cosine instead of poly: ` training\two_stage_warmup_cosine_schedule.py`

## Installation

```bash
python3 -m pip install -r requirements.txt
```

## Data preparation

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**ADE20K**
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```
## Supported Models
| Architecture | Versions                                                                                                                 |
| ------------ | ------------------------------------------------------------------------------------------------------------------------ |
| Linformer    | linformer_vit_tiny_patch16, linformer_vit_small_patch16, linformer_vit_base_patch16, linformer_vit_large_patch16  |
| Hydra        | hydra_vit_tiny_patch16, hydra_vit_small_patch16, hydra_vit_base_patch16, hydra_vit_large_patch16                 |
| Switch       | switch_8_vit_tiny_patch16, switch_8_vit_small_patch16, switch_8_vit_base_patch16, switch_8_vit_large_patch16    |
| Swin         | swin_tiny_window7, swin_wide_tiny_window7, swin_small_window7, swin_wide_small_window7, swin_base_window7, swin_wide_base_window7, swin_large_window7 |
| Synthesizer (FR) | synthesizer_fr_vit_tiny_patch16, synthesizer_fr_vit_small_patch16, synthesizer_fr_vit_base_patch16, synthesizer_fr_vit_large_patch16 |
| NextViT      | nextvit_small_cus, nextvit_base_cus, nextvit_large_cus                                                             |
| ConvViT      | cvt_13, cvt_21, cvt_w24                                                                                            |
| EfficientViT | efficientvit_backbone_b0, efficientvit_backbone_b1, efficientvit_backbone_b2, efficientvit_backbone_b3          |
## Usage

### Training

```

```
