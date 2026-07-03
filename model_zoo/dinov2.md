# EoMT Model Zoo - DINOv2

> FPS measured on NVIDIA H100 with default torch.compile, unless otherwise specified.

## Panoptic Segmentation

### COCO

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
<!-- <tr><td align="left"><a href="../configs/coco/panoptic/eomt_small_640_1x.yaml">EoMT-S</a></td>
<td align="center">640×640</td>
<td align="center">330</td>
<td align="center">44.7</td>
<td align="center">-</td>
</tr> -->
<!-- ROW: EoMT-S 640x640 -->
<tr><td align="left"><a href="../configs/dinov2/coco/panoptic/eomt_small_640_2x.yaml">EoMT-S</a><sup>2x</sup></td>
<td align="center">640×640</td>
<td align="center">330</td>
<td align="center">46.7</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_small_640_2x/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-B 640x640 -->
<!-- <tr><td align="left"><a href="../configs/coco/panoptic/eomt_base_640_1x.yaml">EoMT-B</a></td>
<td align="center">640×640</td>
<td align="center">261</td>
<td align="center">50.6</td>
<td align="center">-</td>
</tr> -->
<!-- ROW: EoMT-B 640x640 -->
<tr><td align="left"><a href="../configs/dinov2/coco/panoptic/eomt_base_640_2x.yaml">EoMT-B</a><sup>2x</sup></td>
<td align="center">640×640</td>
<td align="center">261</td>
<td align="center">51.6</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_base_640_2x/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-L 640x640 -->
<tr><td align="left"><a href="../configs/dinov2/coco/panoptic/eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640×640</td>
<td align="center">128</td>
<td align="center">56.0</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="../configs/dinov2/coco/panoptic/eomt_giant_640.yaml">EoMT-g</a></td>
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
<tr><td align="left"><a href="../configs/dinov2/coco/panoptic/eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">30</td>
<td align="center">58.3</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="../configs/dinov2/coco/panoptic/eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280×1280</td>
<td align="center">12</td>
<td align="center">59.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

### ADE20K

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
<tr><td align="left"><a href="../configs/dinov2/ade20k/panoptic/eomt_large_640.yaml">EoMT-L</a></td>
<td align="center">640×640</td>
<td align="center">128</td>
<td align="center">50.6<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_640/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 640x640 -->
<tr><td align="left"><a href="../configs/dinov2/ade20k/panoptic/eomt_giant_640.yaml">EoMT-g</a></td>
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
<tr><td align="left"><a href="../configs/dinov2/ade20k/panoptic/eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">30</td>
<td align="center">51.7<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
<!-- ROW: EoMT-g 1280x1280 -->
<tr><td align="left"><a href="../configs/dinov2/ade20k/panoptic/eomt_giant_1280.yaml">EoMT-g</a></td>
<td align="center">1280×1280</td>
<td align="center">12</td>
<td align="center">52.8<sup>C</sup></td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_panoptic_eomt_giant_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub><sup>C</sup> Models pre-trained on COCO panoptic segmentation. See above for how to load a checkpoint.</sub>*

## Semantic Segmentation

### Cityscapes

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
<tr><td align="left"><a href="../configs/dinov2/cityscapes/semantic/eomt_large_1024.yaml">EoMT-L</a></td>
<td align="center">1024×1024</td>
<td align="center">25</td>
<td align="center">84.2</td>
<td align="center"><a href="https://huggingface.co/tue-mps/cityscapes_semantic_eomt_large_1024/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

### ADE20K

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
<tr><td align="left"><a href="../configs/dinov2/ade20k/semantic/eomt_large_512.yaml">EoMT-L</a></td>
<td align="center">512×512</td>
<td align="center">92</td>
<td align="center">58.4</td>
<td align="center"><a href="https://huggingface.co/tue-mps/ade20k_semantic_eomt_large_512/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

## Instance Segmentation

### COCO

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
<tr><td align="left"><a href="../configs/dinov2/coco/instance/eomt_large_640.yaml">EoMT-L</a></td>
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
<tr><td align="left"><a href="../configs/dinov2/coco/instance/eomt_large_1280.yaml">EoMT-L</a></td>
<td align="center">1280×1280</td>
<td align="center">30</td>
<td align="center">48.8*</td>
<td align="center"><a href="https://huggingface.co/tue-mps/coco_instance_eomt_large_1280/resolve/main/pytorch_model.bin">Model Weights</a></td>
</tr>
</tbody></table>

*<sub>\* mAP reported using pycocotools; TorchMetrics (used by default) yields ~0.4 lower.</sub>*
