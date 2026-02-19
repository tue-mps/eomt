# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import torch.nn.functional as F

from typing import Optional
import timm
import torch
import math
import os
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn

from models.backbones.utils import vit_sizes
class SynthFR(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="linformer_vit_base_patch16",
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
        # num_patches = (img_size[0] // self.backbone.patch_size) ** 2
        self.backbone.set_image_res(img_size[0])
        self.num_prefix_tokens = self.backbone.num_prefix_tokens
        # for block in self.backbone.blocks:
        #     block.attn.resize(num_patches + self.num_prefix_tokens)
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print("checkpoint found at {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path,weights_only=False)
            
            state_dict = None
            if 'model_state' in checkpoint:
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state'].items()}
                state_dict['pos_embed'] = resize_spatial_weight(
                        state_dict['pos_embed'],
                        self.backbone.pos_embed  
                    )
                model_state = self.backbone.state_dict()
                for key in list(state_dict.keys()): # List to avoid modifying dict while iterating
                    # Check if this is an attention projection layer
                    if "attn" in key and ("proj_k" in key or "proj_v" in key or "random" in key):
                        
                        # Ensure key exists in current model
                        if key in model_state:
                            target_weight = model_state[key]
                            source_weight = state_dict[key]
                            
                            src_len = get_seq_len_for_resize(key,source_weight)
                            tgt_len = get_seq_len_for_resize(key,target_weight)

                            if source_weight.shape != target_weight.shape and src_len != tgt_len:
                                print(f"Resizing attention layer {key}: {source_weight.shape} -> {target_weight.shape}")
                                state_dict[key] = resize_spatial_weight(source_weight, target_weight)
                                        
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False) if state_dict else self.backbone.load_state_dict(checkpoint, strict=False)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
                
         
        self.embed_dim = self.backbone.embed_dim
        sizes = vit_sizes[vit_size]
        self.num_heads = [sizes['num_heads']]
        self.depths = [sizes['depth']]
        self.patch_size = self.backbone.patch_embed.patch_size
        self.grid_size = self.backbone.patch_embed.grid_size
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
    

def get_seq_len_for_resize(key: str, w: torch.Tensor) -> int:
    if w.ndim == 2:
        return w.shape[0]

    if w.ndim == 3:
        if "random_attn_1" in key:
            # [H, L, D] -> seq len is dim=1
            return w.shape[1]
        if "random_attn_2" in key:
            # [H, D, L] -> seq len is dim=2
            return w.shape[2]

        # If you have other 3D tensors, decide what you want here:
        raise ValueError(f"Don't know seq-len dim for 3D tensor key={key}, shape={tuple(w.shape)}")

    raise ValueError(f"Unsupported ndim={w.ndim} for key={key}")

def infer_num_tokens(seq_len_old, seq_len_new):
    """
    Helper to infer the number of extra tokens (like CLS) by checking which 
    subtraction makes the remaining sequence length a perfect square.
    """
    # Try common cases: 0 or 1 extra tokens
    for n in (1, 0):
        # Check if removing n tokens results in a perfect square (e.g. 197-1 = 14*14)
        if (seq_len_old - n) > 0 and int(math.sqrt(seq_len_old - n))**2 == (seq_len_old - n):
            return n
    
    # Fallback: try to infer from the new sequence length
    for n in (1, 0):
        if (seq_len_new - n) > 0 and int(math.sqrt(seq_len_new - n))**2 == (seq_len_new - n):
            return n
            
    # Default to 0 if inference fails
    return 0

def resize_spatial_weight(weight, target_weight, *, align_corners=False):
    """
    Two explicit paths only:

    (A) random_attn resize via 1D linear interpolation over seq_len:
        - r_1 explicit: interpolate(weight.transpose(1,2)).transpose(1,2)   for [H, L, D]
        - r_2 explicit: interpolate(weight)                                for [H, D, L]

    (B) pos_embed / spatial grid resize via bicubic interpolation on square grids.

    F.interpolate: 'linear' expects 3D (N,C,L); 'bicubic' expects 4D (N,C,H,W). [web:20]
    """

    # --------------------------
    # Path A: random_attn resize
    # --------------------------
    if weight.ndim == 3 and target_weight.ndim == 3:
        if weight.shape[0] != target_weight.shape[0]:
            raise ValueError(f"random_attn head dim mismatch: {weight.shape[0]} vs {target_weight.shape[0]}")

        if weight.shape == target_weight.shape:
            return weight.to(dtype=target_weight.dtype, device=target_weight.device)

        w = weight.to(dtype=target_weight.dtype, device=target_weight.device)

        # Explicit r_1: random_attn_1 is [H, L, D] and needs interpolation over L
        # by converting to (N,C,L) = [H, D, L] via transpose, interpolating, then transposing back.
        if w.shape[2] == target_weight.shape[2]:
            new_max_seq_len = target_weight.shape[1]
            r_1 = F.interpolate(
                w.transpose(1, 2),          # [H, D, L_old]
                size=new_max_seq_len,       # L_new
                mode="linear",
                align_corners=align_corners,
            ).transpose(1, 2)               # [H, L_new, D] [web:20]
            return r_1

        # Explicit r_2: random_attn_2 is already [H, D, L], so interpolate directly on last dim.
        if w.shape[1] == target_weight.shape[1]:
            new_max_seq_len = target_weight.shape[2]
            r_2 = F.interpolate(
                w,                          # [H, D, L_old]
                size=new_max_seq_len,       # L_new
                mode="linear",
                align_corners=align_corners,
            )                               # [H, D, L_new] [web:20]
            return r_2

        raise ValueError(
            f"Unrecognized random_attn layout: weight={tuple(weight.shape)} target={tuple(target_weight.shape)}"
        )

    # ---------------------------------------
    # Path B: pos_embed / spatial grid resize
    # ---------------------------------------
    is_2d = (weight.ndim == 2)
    if not (weight.ndim in (2, 3) and target_weight.ndim == weight.ndim):
        raise ValueError(f"Unsupported dims for spatial resize: weight.ndim={weight.ndim}, target.ndim={target_weight.ndim}")

    # Normalize to [1, L, C]
    if is_2d:
        weight_3d = weight.unsqueeze(0)
        target_3d = target_weight.unsqueeze(0)
    else:
        weight_3d = weight
        target_3d = target_weight

    if weight_3d.shape[2] != target_3d.shape[2]:
        raise ValueError(f"Channel dim mismatch: {weight_3d.shape[2]} vs {target_3d.shape[2]}")

    L_old, L_new, C = weight_3d.shape[1], target_3d.shape[1], weight_3d.shape[2]
    if L_old == L_new:
        return weight_3d.squeeze(0) if is_2d else weight_3d

    # infer_num_tokens must exist in your codebase
    num_tokens = infer_num_tokens(L_old, L_new)

    extra_tokens = weight_3d[:, :num_tokens, :]  # [1, T, C]
    grid_tokens  = weight_3d[:, num_tokens:, :]  # [1, G_old, C]
    G_new = L_new - num_tokens

    gs_old = int(math.sqrt(grid_tokens.shape[1]))
    gs_new = int(math.sqrt(G_new))
    if gs_old * gs_old != grid_tokens.shape[1] or gs_new * gs_new != G_new:
        raise ValueError(
            f"Grid tokens not square: G_old={grid_tokens.shape[1]} G_new={G_new} (num_tokens={num_tokens})."
        )

    grid_4d = grid_tokens.reshape(1, gs_old, gs_old, C).permute(0, 3, 1, 2)  # [1,C,H,W]
    grid_4d = F.interpolate(
        grid_4d,
        size=(gs_new, gs_new),
        mode="bicubic",
        align_corners=False,
    )  # [web:20]
    grid_new = grid_4d.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, C)

    out = torch.cat([extra_tokens, grid_new], dim=1)
    return out.squeeze(0) if is_2d else out