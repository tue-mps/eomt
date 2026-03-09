from typing import Optional
from collections import OrderedDict

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.backbones.focalnet import FocalNet


# Maps variant name -> config used when constructing FocalNet directly
focalnet_sizes = {
    "T": {"depths": [2, 2, 6, 2],  "embed_dim": 96,  "focal_levels": [2, 2, 2, 2], "focal_windows": [3, 3, 3, 3]},
    "S": {"depths": [2, 2, 18, 2], "embed_dim": 96,  "focal_levels": [2, 2, 2, 2], "focal_windows": [3, 3, 3, 3]},
    "B": {"depths": [2, 2, 18, 2], "embed_dim": 128, "focal_levels": [2, 2, 2, 2], "focal_windows": [3, 3, 3, 3]},
    "L": {"depths": [2, 2, 18, 2], "embed_dim": 192, "focal_levels": [3, 3, 3, 3], "focal_windows": [5, 5, 5, 5]},
    "XL": {"depths": [2, 2, 18, 2], "embed_dim": 256, "focal_levels": [3, 3, 3, 3], "focal_windows": [5, 5, 5, 5]},
}


class FocalNetModel(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        focalnet_size: str = "T",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
        use_conv_embed: bool = False,
        use_layerscale: bool = False,
        use_postln: bool = False,
        use_postln_in_modulation: bool = False,
        normalize_modulator: bool = False,
    ):
        super().__init__()

        sizes = focalnet_sizes[focalnet_size]

        self.backbone: FocalNet = FocalNet(
            img_size=img_size[0],  # FocalNet expects a single int or square tuple
            in_chans=3,
            num_classes=0,
            embed_dim=sizes["embed_dim"],
            depths=sizes["depths"],
            focal_levels=sizes["focal_levels"],
            focal_windows=sizes["focal_windows"],
            use_conv_embed=use_conv_embed,
            use_layerscale=use_layerscale,
            use_postln=use_postln,
            use_postln_in_modulation=use_postln_in_modulation,
            normalize_modulator=normalize_modulator,
        )

        if ckpt_path is not None:
            self.backbone = load_focalnet_ckpt(self.backbone, ckpt_path)

        self.backbone.set_image_res(img_size[0])

        self.depths = sizes["depths"]
        self.num_heads = [sizes["embed_dim"] * (2 ** i) // 32 for i in range(len(self.depths))]
        self.attn_multiplier = multiplier

        # embed_dim of the final stage (used by EoMT for query dimensionality)
        self.embed_dim = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size          # (ph, pw)
        self.num_prefix_tokens = 0

        # grid size at the final stage (after all downsampling)
        num_stages = len(self.depths)
        patches_res = self.backbone.patches_resolution
        self.grid_size = (
            patches_res[0] // (2 ** (num_stages - 1)),
            patches_res[1] // (2 ** (num_stages - 1)),
        )

        # ----------------------------------------------------------------
        # Flatten all FocalNetBlocks into a single list (mirrors Swin/Hydra)
        # Each BasicLayer holds its own nn.ModuleList of FocalNetBlocks.
        # ----------------------------------------------------------------
        self.blocks = [block for layer in self.backbone.layers for block in layer.blocks]

        # cumulative block-end index per stage (1-based), for downsample tracking
        depth_prefix_sum_list = []
        s = 0
        for d in self.depths:
            s += d
            depth_prefix_sum_list.append(s)

        # Collect the PatchEmbed downsamplers that sit at the end of each stage
        # (BasicLayer.downsample is a PatchEmbed instance, None for the last stage)
        self.patch_embed_dict = nn.ModuleDict(
            {
                str(end): layer.downsample
                for end, layer in zip(depth_prefix_sum_list, self.backbone.layers)
                if layer.downsample is not None
            }
        )

        self.norm = self.backbone.norm

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    # ------------------------------------------------------------------
    # EoMT interface — mirrors Swin / Hydra contracts exactly
    # ------------------------------------------------------------------

    def pre_block(self, x: torch.Tensor):
        """Patch-embed input image into flat token sequence (B, H*W, C)."""
        x, H, W = self.backbone.patch_embed(x)
        x = self.backbone.pos_drop(x)
        # Store spatial dims so run_block can pass them to each FocalNetBlock
        self._H, self._W = H, W
        return x

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        """
        EoMT-aware FocalNet block runner.

        FocalNetBlock.forward() expects (B, H*W, C) and uses block.H / block.W
        to reshape internally. We set those before calling the block.

        x  : (B, L, C)  — flat spatial tokens
        q  : (num_q, C) — unbatched query tokens, consistent with EoMT.forward()
        i  : global block index (0-based across all stages)
        """
        H, W = self._H, self._W
        B, L, C = x.shape
        assert L == H * W, (
            f"Token count {L} does not match spatial dims {H}x{W}={H * W}"
        )

        block.H, block.W = H, W

        if i >= len(self.blocks) - eomt_obj.num_blocks:
            eomt_idx = i - (len(self.blocks) - eomt_obj.num_blocks)

            # Expand q to batch dim
            if q.dim() == 2:
                q_batched = q[None].expand(B, -1, -1)
            else:
                q_batched = q

            # 1) Concatenate query + image tokens: (B, num_q+L, C)
            xq = torch.cat([q_batched, x], dim=1)

            # 2) Pre-attention norm (FocalNetBlock uses norm1 before modulation)
            pre_attn = block.norm1(xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]   # (B, num_q, C)
            x_tok = pre_attn[:, eomt_obj.num_q :, :]   # (B, L, C)

            # 3) EoMT prediction heads
            mask_logits, class_logits = eomt_obj._predict(x_tok, q_tok)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)

            # 4) EoMT cross-attention
            after_eomt = torch.cat([q_tok, x_tok], dim=1)  # (B, num_q+L, C)
            new_x = eomt_obj.attn[eomt_idx](self.norm(after_eomt))
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[eomt_idx](new_x))

            # 5) Split back
            q_batched = xq[:, : eomt_obj.num_q, :]   # (B, num_q, C)
            x = xq[:, eomt_obj.num_q :, :]            # (B, L, C)

            # 6) Standard FocalNet modulation block on image tokens
            x = block(x)

            # 7) Mirror FFN sub-layer for query tokens (no self-modulation)
            #    FocalNetBlock: shortcut + drop_path(gamma_2 * mlp(norm2(q)))
            q_batched = q_batched + block.drop_path(
                block.gamma_2 * (
                    block.norm2(block.mlp(q_batched))
                    if block.use_postln
                    else block.mlp(block.norm2(q_batched))
                )
            )

            # Collapse batch dim back to (num_q, C)
            q = q_batched

        else:
            # Plain FocalNet block
            x = block(x)

        # Downsample at stage boundary: PatchEmbed expects (B, C, H, W)
        ds_idx = str(i + 1)
        if ds_idx in self.patch_embed_dict:
            x_2d = x.transpose(1, 2).reshape(B, C, H, W)
            x, H, W = self.patch_embed_dict[ds_idx](x_2d)
            self._H, self._W = H, W

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        """Apply final backbone norm after all blocks."""
        x = self.norm(x)
        return x, q


def load_focalnet_ckpt(model: FocalNet, ckpt_path: str) -> FocalNet:
    """Load a FocalNet checkpoint, tolerating key mismatches gracefully."""
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)

    if "model" in checkpoint:
        state_dict = checkpoint["model"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    elif "model_state" in checkpoint:
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in checkpoint["model_state"].items()
        }
    else:
        state_dict = checkpoint

    # Strip "backbone." prefix if present (e.g. from segmentation checkpoints)
    state_dict = {
        k[9:] if k.startswith("backbone.") else k: v
        for k, v in state_dict.items()
    }

    # Remove classification head weights — not used for segmentation
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("FocalNet — Missing keys :", missing)
    print("FocalNet — Unexpected keys:", unexpected)
    return model
