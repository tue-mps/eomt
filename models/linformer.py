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


class Linformer(nn.Module):
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
        self.backbone.set_image_res()
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
                    if "attn" in key and ("proj_k" in key or "proj_v" in key):
                        
                        # Ensure key exists in current model
                        if key in model_state:
                            target_weight = model_state[key]
                            source_weight = state_dict[key]
                            
                            # Check for shape mismatch in sequence length
                            # Note: Checking flattened length (L_old vs L_new)
                            # 2D case: shape[0] is length. 3D case: shape[1] is length.
                            src_len = source_weight.shape[0] if source_weight.ndim == 2 else source_weight.shape[1]
                            tgt_len = target_weight.shape[0] if target_weight.ndim == 2 else target_weight.shape[1]

                            if source_weight.shape != target_weight.shape and src_len != tgt_len:
                                print(f"Resizing attention layer {key}: {source_weight.shape} -> {target_weight.shape}")
                                
                                state_dict[key] = resize_spatial_weight(
                                    source_weight,
                                    target_weight
                                )
            
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

def resize_spatial_weight(weight, target_weight):
    """
    Universal resizer for both Position Embeddings and Attention Projections.
    
    Args:
        weight (Tensor): Source weight from checkpoint. 
                         Shape: [1, L, C] (pos_embed) OR [L, C] (attn proj)
        target_weight (Tensor): Target weight from current model.
    
    Returns:
        Tensor: Resized weight matching target_weight shape.
    """
    # 1. Capture original rank to restore it later
    is_2d = (weight.ndim == 2)
    
    # 2. Normalize to 3D: [1, L, C]
    # If [L, C], unsqueeze to [1, L, C]
    if is_2d:
        weight = weight.unsqueeze(0)
        target_shape = target_weight.unsqueeze(0).shape
    else:
        target_shape = target_weight.shape

    L_old = weight.shape[1]
    L_new = target_shape[1]
    C = weight.shape[2]

    if L_old == L_new:
        return weight.squeeze(0) if is_2d else weight

    # 3. Infer extra tokens (CLS, Distillation, etc.)
    num_tokens = infer_num_tokens(L_old, L_new)

    # 4. Separate Extra tokens vs Grid tokens
    extra_tokens = weight[:, :num_tokens, :]      # [1, num_tokens, C]
    grid_tokens = weight[:, num_tokens:, :]       # [1, grid_old, C]

    # 5. Calculate spatial dimensions (H, W)
    gs_old = int(math.sqrt(grid_tokens.shape[1])) # e.g., 14
    gs_new = int(math.sqrt(L_new - num_tokens))   # e.g., 32

    # 6. Reshape for Bicubic Interpolation
    # [1, L_grid, C] -> [1, H, W, C] -> [1, C, H, W]
    grid_tokens = grid_tokens.reshape(1, gs_old, gs_old, C).permute(0, 3, 1, 2)

    # 7. Interpolate
    grid_tokens = F.interpolate(
        grid_tokens,
        size=(gs_new, gs_new),
        mode='bicubic',
        align_corners=False
    )

    # 8. Reshape back
    # [1, C, H, W] -> [1, H, W, C] -> [1, L_grid_new, C]
    grid_tokens = grid_tokens.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, C)

    # 9. Concatenate
    new_weight = torch.cat([extra_tokens, grid_tokens], dim=1)

    # 10. Restore original rank if necessary
    if is_2d:
        return new_weight.squeeze(0)
    
    return new_weight