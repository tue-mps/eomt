# Your ViT is Secretly an Image Segmentation Model (CVPR 2025)

*[Tommie Kerssies](https://tommiekerssies.com)¹, [Niccolò Cavagnero](https://scholar.google.com/citations?user=Pr4XHRAAAAAJ)²<sup>,</sup>\*, [Alexander Hermans](https://scholar.google.de/citations?user=V0iMeYsAAAAJ)³, [Narges Norouzi](https://scholar.google.com/citations?user=q7sm490AAAAJ)¹, [Giuseppe Averta](https://www.giuseppeaverta.me/)², [Bastian Leibe](https://scholar.google.com/citations?user=ZcULDB0AAAAJ)³, [Gijs Dubbelman](https://scholar.google.nl/citations?user=wy57br8AAAAJ)¹, [Daan de Geus](https://ddegeus.github.io)¹<sup>,</sup>³*

<sup>1</sup> Eindhoven University of Technology  
<sup>2</sup> Polytechnic of Turin  
<sup>3</sup> RWTH Aachen University  
<sup>\*</sup> _Work done while visiting RWTH Aachen University._


---

Welcome to the official repository for "**Your ViT is Secretly an Image Segmentation Model**".

📄 **Paper**: [arXiv](https://arxiv.org/abs/2503.19108)  
👁️ **Project page**: [https://tue-mps.github.io/eomt](https://tue-mps.github.io/eomt)  
🛎️ **Stay updated**: [Watch the repository](https://github.com/tue-mps/eomt/subscription)  
🐞 **Questions or issues?** [Open a GitHub issue](https://github.com/tue-mps/eomt/issues)  
📬 **Contact**: t.kerssies[at]tue[dot]nl

---

## Overview

We present the **Encoder-only Mask Transformer** (EoMT), a minimalist image segmentation model built solely on a plain Vision Transformer. No adapters, no decoders, just the ViT.

Leveraging large, extensively pre-trained ViTs, EoMT achieves accuracy similar to state-of-the-art methods that rely on complex, task-specific components. At the same time, it is significantly faster thanks to its simplicity, e.g., up to 4× faster with ViT-L.  

Turns out, *your ViT is secretly an image segmentation model*. EoMT demonstrates that architectural complexity isn’t necessary, plain Transformer power is all you need.

---

## Installation

If you don't have Conda installed, install Miniconda and restart your shell:

```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Then create the environment, activate it, and install the dependencies:

```bash
conda create -n EoMT python==3.12
conda activate EoMT
python3 -m pip install -r requirements.txt
```

---

## Weights & Biases

[Weights & Biases](https://wandb.ai/) (wandb) is used for experiment logging and visualization. To enable wandb, log in to your account:

```bash
wandb login
```

---

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

---

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

This trains the `EoMT-L` model on COCO panoptic segmentation with a 640×640 input size using 4 GPUs. Each GPU processes a batch of 4 images, for a total batch size of 16.  

✅ The total batch size must be `devices × batch_size = 16`  
🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.

> This configuration takes ~6 hours on 4×NVIDIA H100 GPUs, each using ~26GB VRAM.

To fine-tune a pre-trained EoMT model, add:

```bash
  --model.load_ckpt_class_head False
  --model.ckpt_path /path/to/pytorch_model.bin \
```

🔧 Replace `/path/to/pytorch_model.bin` with the checkpoint to fine-tune.  

> `--model.load_ckpt_class_head False` skips loading the class prediction head when fine-tuning on a dataset with different categories.  

### Evaluating

To evaluate a pre-trained EoMT model, run:

```bash
python3 main.py validate \
  -c configs/coco/panoptic/eomt_large_640.yaml \
  --trainer.devices 4 \
  --data.batch_size 4 \
  --data.path /path/to/dataset \
  --model.ckpt_path /path/to/pytorch_model.bin
```

This evaluates the same `EoMT-L` model using 4 GPUs with a batch size of 4 per GPU.

🔧 Replace `/path/to/dataset` with the directory containing the dataset zip files.  
🔧 Replace `/path/to/pytorch_model.bin` with the checkpoint to evaluate.

A [notebook](inference.ipynb) is also available for quick inference and visualization with pre-trained models.

---

## Model Zoo

> All FPS values were measured on an NVIDIA H100 GPU.

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
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">56.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">58.3</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_giant_640.yaml">EoMT-g</a></td>
<td align="center">640x640</td>
<td align="center">55</td>
<td align="center">57.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="configs/coco_panoptic_eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280x1280</td>
<td align="center">12</td>
<td align="center">59.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

#### ADE20K

> ADE20K panoptic results use COCO pre-training. See above for how to load a checkpoint.

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
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">50.6</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">51.7</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_giant_640.yaml">EoMT-g</a></td>
<td align="center">640x640</td>
<td align="center">55</td>
<td align="center">51.3</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="configs/ade20k_panoptic_eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280x1280</td>
<td align="center">12</td>
<td align="center">52.8</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

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
<tr><td align="left"><a href="configs/cityscapes_semantic_eomt_large_1024.yaml">EoMT-L</a></td>
<td align="center">1024x1024</td>
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
<tr><td align="left"><a href="configs/ade20k_semantic_eomt_large_512.yaml">EoMT-L</a></td>
<td align="center">512x512</td>
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
<tr><td align="left"><a href="configs/coco_instance_eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640x640</td>
<td align="center">128</td>
<td align="center">45.2*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 1280x1280 -->
<tr><td align="left"><a href="configs/coco_instance_eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280x1280</td>
<td align="center">30</td>
<td align="center">48.8*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub>\*(mAP reported using pycocotools; TorchMetrics (used by default) yields ~0.7 lower)</sub>*

---

## Citation
If you find this work useful in your research, please cite it using the BibTeX entry below:

```BibTeX
@inproceedings{kerssies2025eomt,
  author    = {Kerssies, Tommie and Cavagnero, Niccolò and Hermans, Alexander and Norouzi, Narges and Averta, Giuseppe and Leibe, Bastian and Dubbelman, Gijs and de Geus, Daan},
  title     = {Your ViT is Secretly an Image Segmentation Model},
  booktitle = {IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2025},
}
```

---

## Acknowledgements

This project builds upon code from the following libraries and repositories:

- [Hugging Face Transformers](https://github.com/huggingface/transformers) (Apache-2.0 License)  
- [PyTorch Image Models (timm)](https://github.com/huggingface/pytorch-image-models) (Apache-2.0 License)  
- [PyTorch Lightning](https://github.com/Lightning-AI/pytorch-lightning) (Apache-2.0 License)  
- [TorchMetrics](https://github.com/Lightning-AI/torchmetrics) (Apache-2.0 License)  
- [Mask2Former](https://github.com/facebookresearch/Mask2Former) (Apache-2.0 License)
- [Detectron2](https://github.com/facebookresearch/detectron2) (Apache-2.0 License)