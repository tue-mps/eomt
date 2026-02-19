# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import torch.nn.functional as F

from typing import Optional
import timm
import torch
import os
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn

from models.backbones.utils import vit_sizes

class Switch(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="vit_large_patch16_384",
        multiplier: int = 1,
        ls_init = 1e-6,
        vit_size: str = "B",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size[0],
            no_embed_class = True,
            init_values=ls_init,
            num_classes=0
        )
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print("checkpoint found at {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path,weights_only=False)
            
            state_dict = None
            if 'model_state' in checkpoint:
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state'].items()}
                state_dict['pos_embed'] = resize_pos_embed(
                state_dict['pos_embed'],
                self.backbone.pos_embed)
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False) if state_dict else self.backbone.load_state_dict(checkpoint, strict=False)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
                
         
        self.embed_dim = self.backbone.embed_dim
        sizes = vit_sizes[vit_size]
        self.num_heads = [sizes['num_heads']]
        self.depths = [sizes['depth']]
        self.patch_size = self.backbone.patch_embed.patch_size
        self.grid_size = self.backbone.patch_embed.grid_size
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        self.blocks = self.backbone.blocks
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
        x = self.backbone._pos_embed(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        return x
    
    def q_mlp(self, block, q: torch.Tensor):
        q = q + block.drop_path2(block.ls2(block.mlp(block.norm2(q))))
        return q
    
    def run_block(self, block, x: torch.Tensor, _):
        x = block(x)
        return x
    


def resize_pos_embed(posemb, posemb_new):
    # posemb: [1, L_old, C], posemb_new: [1, L_new, C]
    L_old = posemb.shape[1]
    L_new = posemb_new.shape[1]

    if L_old == L_new:
        return posemb

    # infer how many extra tokens (e.g. CLS) there are
    num_tokens = infer_num_tokens(posemb, posemb_new)

    extra_tokens = posemb[:, :num_tokens]
    posemb_grid = posemb[:, num_tokens:]          # [1, old_grid, C]

    gs_old = int((posemb_grid.shape[1]) ** 0.5)   # e.g. 14 for 196 tokens
    gs_new = int((L_new - num_tokens) ** 0.5)     # e.g. 32 for 1024 tokens

    posemb_grid = posemb_grid.reshape(1, gs_old, gs_old, -1).permute(0, 3, 1, 2)
    posemb_grid = F.interpolate(posemb_grid, size=(gs_new, gs_new),
                                mode='bicubic', align_corners=False)
    posemb_grid = posemb_grid.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, -1)
    print(posemb_grid.shape, extra_tokens.shape)
    return torch.cat([extra_tokens, posemb_grid], dim=1)

def infer_num_tokens(posemb_ckpt, posemb_model):
    # posemb_ckpt: [1, L_ckpt, C]; posemb_model: [1, L_model, C]
    Lc = posemb_ckpt.shape[1]
    Lm = posemb_model.shape[1]

    # Try common cases: 0 or 1 extra tokens
    for n in (1, 0):
        if (Lc - n) > 0 and int((Lc - n) ** 0.5) ** 2 == (Lc - n):
            return n
    # Fallback: assume same #extra as model
    for n in (1, 0):
        if (Lm - n) > 0 and int((Lm - n) ** 0.5) ** 2 == (Lm - n):
            return n
    raise ValueError(f"Cannot infer num_tokens from shapes {posemb_ckpt.shape} and {posemb_model.shape}")