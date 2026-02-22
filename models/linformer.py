# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

import torch.nn.functional as F
from typing import Optional
import timm
import torch
import math
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import os
import torch.nn as nn

from models.backbones.utils import vit_sizes


class Linformer(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="linformer_vit_base_patch16",
        multiplier: int = 1,
        ls_init: float = 1e-6,
        vit_size: str = "B",
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        # NOTE: original file uses img_size[0] (square assumption) for timm linformer.
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size[0],
            no_embed_class=True,
            init_values=ls_init,
            num_classes=0,
        )

        # Keep existing behavior (even though set_image_res is a no-op in some backbones).
        self.backbone.set_image_res()

        self.num_prefix_tokens = self.backbone.num_prefix_tokens

        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print("checkpoint found at {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path,weights_only=False)
            
            state_dict = None
            if 'model_state' in checkpoint:
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state'].items()}
                print(f"Resizing position embeddings pos_embed: {state_dict['pos_embed'].shape} -> {self.backbone.pos_embed .shape}")
                state_dict['pos_embed'] = resize_spatial_weight(
                        state_dict['pos_embed'],
                        self.backbone.pos_embed
                    )
                model_state = self.backbone.state_dict()
                for key in list(state_dict.keys()):
                    if "attn" in key and ("proj_k" in key or "proj_v" in key):
                        if key in model_state:
                            target_weight = model_state[key]
                            source_weight = state_dict[key]
                            
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

        # ---- expose fields in same style as vit.py ----
        self.embed_dim = self.backbone.embed_dim
        sizes = vit_sizes[vit_size]
        self.num_heads = [sizes["num_heads"]]
        self.depths = [sizes["depth"]]
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

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        """
        Mirrors models/vit.py run_block signature + EoMT workflow.
        Assumes x is [B, N, C] tokens (as in ViT/Linformer timm models).
        """
        if i >= len(self.blocks) - eomt_obj.num_blocks:
            # ---- EoMT path (same as vit.py) ----
            xq = torch.cat((q[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)
            pre_attn = block.norm1(xq)
            x, q = pre_attn[:, eomt_obj.num_q :, :], pre_attn[:, : eomt_obj.num_q, :]

            mask_logits, class_logits = eomt_obj._predict(x, q)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)

            after_eomt = torch.cat((q, x), dim=1)

            # Cross-attention with EoMT attention module (same call pattern as vit.py).
            new_x = eomt_obj.attn[i - len(self.blocks)](self.norm(after_eomt))
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[i - len(self.blocks)](new_x))
            x, q = xq[:, eomt_obj.num_q :, :], xq[:, : eomt_obj.num_q, :]

            # Then apply the block’s own attention/mlp residual paths (same as vit.py).
            x = x + block.drop_path1(block.ls1(block.attn(block.norm1(x))))
            x = x + block.drop_path2(block.ls2(block.mlp(block.norm2(x))))
            q = q + block.drop_path2(block.ls2(block.mlp(block.norm2(q))))
        else:
            x = block(x)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        # For Linformer/ViT token tensors this is already [B, N, C], so LayerNorm is valid
        x = self.norm(x)
        return x, q


def infer_num_tokens(seq_len_old: int, seq_len_new: int) -> int:
    # Try common cases: 0 or 1 extra tokens
    for n in (1, 0):
        if (seq_len_old - n) > 0 and int(math.sqrt(seq_len_old - n)) ** 2 == (seq_len_old - n):
            return n
    for n in (1, 0):
        if (seq_len_new - n) > 0 and int(math.sqrt(seq_len_new - n)) ** 2 == (seq_len_new - n):
            return n
    return 0


def resize_spatial_weight(weight: torch.Tensor, target_weight: torch.Tensor) -> torch.Tensor:
    """
    Universal resizer for both Position Embeddings and Attention Projections.
    Supports:
      - pos_embed: [1, L, C]
      - attn proj: [L, C]  (will be temporarily treated as [1, L, C])
    """
    is_2d = (weight.ndim == 2)

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

    num_tokens = infer_num_tokens(L_old, L_new)

    extra_tokens = weight[:, :num_tokens, :]
    grid_tokens = weight[:, num_tokens:, :]

    gs_old = int(math.sqrt(grid_tokens.shape[1]))
    gs_new = int(math.sqrt(L_new - num_tokens))

    grid_tokens = grid_tokens.reshape(1, gs_old, gs_old, C).permute(0, 3, 1, 2)
    grid_tokens = F.interpolate(grid_tokens, size=(gs_new, gs_new), mode="bicubic", align_corners=False)
    grid_tokens = grid_tokens.permute(0, 2, 3, 1).reshape(1, gs_new * gs_new, C)

    new_weight = torch.cat([extra_tokens, grid_tokens], dim=1)
    return new_weight.squeeze(0) if is_2d else new_weight
