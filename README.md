# Your ViT is Secretly an Image Segmentation Model  
**CVPR 2025 ✨ Highlight** · [📄 Paper](https://arxiv.org/abs/2503.19108)

**[Tommie Kerssies](https://tommiekerssies.com)<sup>1</sup>, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)<sup>2,*</sup>, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)<sup>3</sup>, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)<sup>1</sup>, [Giuseppe Averta](https://www.giuseppeaverta.me/)<sup>2</sup>, [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)<sup>3</sup>, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)<sup>1</sup>, [Daan de Geus](https://ddegeus.github.io)<sup>1,3</sup>**

¹ Eindhoven University of Technology  
² Polytechnic of Turin  
³ RWTH Aachen University  
\* Work done while visiting RWTH Aachen University

## Overview

We present the **Encoder-only Mask Transformer (EoMT)**, a minimalist image segmentation model that repurposes a plain Vision Transformer (ViT) to jointly encode image patches and segmentation queries as tokens. No adapters. No decoders. Just the ViT.

Leveraging large-scale pre-trained ViTs, EoMT achieves accuracy similar to state-of-the-art methods that rely on complex, task-specific components. At the same time, it is significantly faster thanks to its simplicity, for example up to 4× faster with ViT-L.  

Turns out, *your ViT is secretly an image segmentation model*. EoMT shows that architectural complexity isn't necessary. For segmentation, a plain Transformer is all you need.

## News

- [2025-08-15]: 🚀 **EoMT is supported in [LightlyTrain](https://github.com/lightly-ai/lightly-train)!**  
  Pre-train ViT backbones and fine-tune with EoMT in just [a few lines of code](https://docs.lightly.ai/train/stable/semantic_segmentation.html). LightlyTrain is compatible with DINOv3 models. 🚀

## 🤗 Transformers Quick Usage Example

EoMT is also available on [Hugging Face Transformers](https://huggingface.co/docs/transformers/main/model_doc/eomt). To install:

```bash
pip install transformers
```

You can use EoMT for segmentation in just a few lines of code. Replace `model_id` with any model from [this list](https://huggingface.co/models?library=transformers&other=eomt&sort=trending):

```python
import torch
from PIL import Image
import requests
import matplotlib.pyplot as plt
from transformers import EomtForUniversalSegmentation, AutoImageProcessor

model_id = "tue-mps/coco_panoptic_eomt_large_640"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)

image = Image.open(requests.get(
    "http://images.cocodataset.org/val2017/000000039769.jpg", stream=True).raw)

inputs = processor(images=image, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)

original_image_sizes = [(image.height, image.width)]
preds = processor.post_process_panoptic_segmentation(outputs, original_image_sizes)

plt.imshow(preds[0]["segmentation"])
plt.axis("off")
plt.title("Panoptic Segmentation")
plt.show()
```

## Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n eomt python==3.13.2
conda activate eomt
python3 -m pip install -r requirements.txt
```

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

## Data preparation

Download the datasets below depending on which datasets you plan to use.  
You do **not** need to unzip any of the downloaded files.  
Simply place them in a directory of your choice and provide that path via the `--data.path` argument.  
The code will read the `.zip` files directly.

**COCO**
```bash
wget http://images.cocodataset.org/zips/train2017.zip
wget http://images.cocodataset.org/zips/val2017.zip
wget http://images.cocodataset.org/annotations/annotations_trainval2017.zip
wget http://images.cocodataset.org/annotations/panoptic_annotations_trainval2017.zip
```

**ADE20K**
```bash
wget http://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip
wget http://sceneparsing.csail.mit.edu/data/ChallengeData2017/annotations_instance.tar
tar -xf annotations_instance.tar
zip -r -0 annotations_instance.zip annotations_instance/
rm -rf annotations_instance.tar
rm -rf annotations_instance
```

**Cityscapes**
```bash
wget --keep-session-cookies --save-cookies=cookies.txt --post-data 'username=<your_username>&password=<your_password>&submit=Login' https://www.cityscapes-dataset.com/login/
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=1
wget --load-cookies cookies.txt --content-disposition https://www.cityscapes-dataset.com/file-handling/?packageID=3
```

🔧 Replace `<your_username>` and `<your_password>` with your actual [Cityscapes](https://www.cityscapes-dataset.com/) login credentials.  

## Usage

### Training

To train EoMT from scratch, run:

```bash
python3 main.py fit \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset
```

This command trains the `EoMT-L` model with a 640×640 input size on COCO panoptic segmentation using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16.  

✅ Make sure the total batch size is `devices × batch_size = 16`  
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

> This configuration takes ~6 hours on 4×NVIDIA H100 GPUs, each using ~26GB VRAM.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.ckpt_path /path/to/pytorch_model.bin \
  --model.load_ckpt_class_head False
```

🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to fine-tune.  
> `--model.load_ckpt_class_head False` skips loading the classification head when fine-tuning on a dataset with different classes. 

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --model.network.masked_attn_enabled False \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This command evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the path to the checkpoint to evaluate.

A [notebook](inference.ipynb) is available for quick inference and visualization with auto-downloaded pre-trained models.

## Model Zoo

> FPS measured on NVIDIA H100, unless otherwise specified.

### Panoptic Segmentation

#### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-S 640x640 -->
<!-- <tr><td align="left"><a href="configs/coco/panoptic/eomt_small_640_1x.yaml">EoMT-S</a></td>
<td align="center">640×640</td>
<td align="center">330</td>
<td align="center">44.7</td>
<td align="center">-</td>
</tr> -->
<!-- ROW: EoMT-S 640x640 -->
<tr><td align="left"><a href="configs/coco/panoptic/eomt_small_640_2x.yaml">EoMT-S</a><sup>2x</sup></td>
<td align="center">640×640</td>
<td align="center">330</td>
<td align="center">46.7</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-B 640x640 -->
<!-- <tr><td align="left"><a href="configs/coco/panoptic/eomt_base_640_1x.yaml">EoMT-B</a></td>
<td align="center">640×640</td>
<td align="center">261</td>
<td align="center">50.6</td>
<td align="center">-</td>
</tr> -->
<!-- ROW: EoMT-B 640x640 -->
<tr><td align="left"><a href="configs/coco/panoptic/eomt_base_640_2x.yaml">EoMT-B</a><sup>2x</sup></td>
<td align="center">640×640</td>
<td align="center">261</td>
<td align="center">51.6</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_base_640_2x/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco/panoptic/eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640×640</td>
<td align="center">128</td>
<td align="center">56.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="configs/coco/panoptic/eomt_giant_640.yaml">EoMT-g</a></td>
<td align="center">640×640</td>
<td align="center">55</td>
<td align="center">57.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<tr>
  <td align="left"><a href="https://huggingface.co/facebook/webssl-dino7b-full8b-518">EoMT-7B</a></td>
  <td align="center">640×640</td>
  <td align="center">32*</td>
  <td align="center">58.4</td>
  <td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_7b_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<tr>
  <td align="left"><em>ViT-Adapter-7B + M2F</em></td>
  <td align="center"><em>640×640</em></td>
  <td align="center"><em>17*</em></td>
  <td align="center"><em>58.4</em></td>
  <td align="center"><em>-</em></td>
</tr>
</tbody></table>  

*<sup><sup>2x</sup> Longer training schedule. \* FPS measured on NVIDIA B200.</sup>*  

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco/panoptic/eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">30</td>
<td align="center">58.3</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="configs/coco/panoptic/eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280×1280</td>
<td align="center">12</td>
<td align="center">59.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

#### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/ade20k/panoptic/eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640×640</td>
<td align="center">128</td>
<td align="center">50.6<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="configs/ade20k/panoptic/eomt_giant_640.yaml">EoMT-g</a></td>
<td align="center">640×640</td>
<td align="center">55</td>
<td align="center">51.3<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">PQ</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/ade20k/panoptic/eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">30</td>
<td align="center">51.7<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="configs/ade20k/panoptic/eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280×1280</td>
<td align="center">12</td>
<td align="center">52.8<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub><sup>C</sup> Models pre-trained on COCO panoptic segmentation. See above for how to load a checkpoint.</sub>*

### Semantic Segmentation

#### Cityscapes

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 1024x1024 -->
<tr><td align="left"><a href="configs/cityscapes/semantic/eomt_large_1024.yaml">EoMT-L</a></td>
<td align="center">1024×1024</td>
<td align="center">25</td>
<td align="center">84.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/cityscapes_semantic_eomt_large_1024/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

#### ADE20K

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mIoU</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 512x512 -->
<tr><td align="left"><a href="configs/ade20k/semantic/eomt_large_512.yaml">EoMT-L</a></td>
<td align="center">512×512</td>
<td align="center">92</td>
<td align="center">58.4</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

### Instance Segmentation

#### COCO

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mAP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco/instance/eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640×640</td>
<td align="center">128</td>
<td align="center">45.2*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

<table><tbody>
<!-- START TABLE -->
<!-- TABLE HEADER -->
<th valign="bottom">Config</th>
<th valign="bottom">Input size</th>
<th valign="bottom">FPS</th>
<th valign="bottom">mAP</th>
<th valign="bottom">Download</th>
<!-- TABLE BODY -->
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco/instance/eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">30</td>
<td align="center">48.8*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub>\* mAP reported using pycocotools; TorchMetrics (used by default) yields ~0.7 lower.</sub>*

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccol\`{o} and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and {de Geus}, Daan},
  title     = {{Your ViT is Secretly an Image Segmentation Model}},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)
