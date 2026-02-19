

from typing import Optional

from collections import OrderedDict
import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn
from models.backbones.swin import SwinTransformer,swin_sizes


class Swin(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="swin_large_patch4_window7_224", # patch_size=4, window_size=7, embed_dim=96, depths=(2, 2, 18, 2), num_heads=(3, 6, 12, 24)
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
        swin_size: str = "Ti"
    ):
        super().__init__()
        self.backbone: SwinTransformer = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            num_classes=0
        )
        self.backbone.set_image_res(img_size)
        self.backbone = load_swin_ckpt_ignore_attn_mask(self.backbone, ckpt_path)
        sizes = swin_sizes[swin_size]
        self.num_heads = sizes['num_heads']
        self.embed_dim = self.backbone.embed_dim * 2**(len(self.num_heads)-1)
        self.patch_size = self.backbone.patch_embed.patch_size
        self.grid_size = tuple(x // (2**(len(self.num_heads) - 1)) for x in self.backbone.patch_embed.patches_resolution)    
        self.num_prefix_tokens = 0
        self.depths = sizes['depths']
        depth_prefix_sum_list = [0]
        for d in self.depths:
            depth_prefix_sum_list.append(depth_prefix_sum_list[-1]+d)
        depth_prefix_sum_list.pop(0)
        self.patch_merging_dict = nn.ModuleDict({str(d):layer.downsample for d,layer in zip(depth_prefix_sum_list,self.backbone.layers)})
        self.blocks = [block for layer in self.backbone.layers for block in layer.blocks]
        self.norm = self.backbone.norm
        self.attn_multiplier = multiplier
        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
    
    def pre_block(self, x: torch.Tensor):
        x = self.backbone.patch_embed(x)
        return x
    
    def q_mlp(self, block, q: torch.Tensor):
        q = q + block.drop_path(block.mlp(block.norm2(q)))
        return q
    
    def run_block(self, block, x: torch.Tensor,  downsample_index: int):
        x = block(x)
        ds_idx = str(downsample_index+1)
        if ds_idx in self.patch_merging_dict.keys() and self.patch_merging_dict[ds_idx] is not None:
            x = self.patch_merging_dict[ds_idx](x)
        return x
    
def load_swin_ckpt_ignore_attn_mask(model, ckpt_path):
    # 1) Load checkpoint state dict
    checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    if 'model_state' in checkpoint:
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state'].items()}
    elif 'teacher' in checkpoint:
        state_dict = {k[9:] if k.startswith('backbone.') else k: v for k, v in checkpoint['teacher'].items()}
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = OrderedDict()

    # 2) Build a reference of current model shapes for conditional filtering
    model_state = model.state_dict()

    for k, v in state_dict.items():
        if "attn_mask" in k and k in model_state:
            # Condition on image-size–dependent mismatch:
            # if shape doesn't match current model buffer, skip it
            if v.shape != model_state[k].shape:
                print(f"Skipping {k}: ckpt {tuple(v.shape)} != model {tuple(model_state[k].shape)}")
                continue  # ignore this key, mask will be recomputed at runtime

        new_state_dict[k] = v

    # 3) Load with strict=False to tolerate any remaining non-critical diffs
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model