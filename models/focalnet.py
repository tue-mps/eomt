from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Import the backbone module so its @register_model decorators run,
# registering focalnet_* names into timm's model registry.
import models.backbones.focalnet  # noqa: F401


# Structural metadata for the small SRF variant.
# depths/embed_dim must match what focalnet_small_srf registers.
focalnet_sizes = {
    "S": {
        "backbone_name": "focalnet_small_srf",
        "depths": [2, 2, 18, 2],
        "embed_dim": 96,
        "num_heads": [3, 6, 12, 24],  # embed_dim * 2^i / 32
    },
}


class FocalNet(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "focalnet_small_srf",
        focalnet_size: str = "S",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size[0],
            num_classes=0,
        )
        self.backbone.set_image_res(img_size[0])

        if ckpt_path is not None:
            self.backbone = load_focalnet_ckpt(self.backbone, ckpt_path)

        sizes = focalnet_sizes[focalnet_size]
        self.depths = sizes["depths"]
        self.num_heads = sizes["num_heads"]
        self.attn_multiplier = multiplier

        # embed_dim of the final stage — used by EoMT for query dimensionality
        self.embed_dim = self.backbone.num_features
        self.patch_size = self.backbone.patch_embed.patch_size
        self.num_prefix_tokens = 0

        # grid size after all downsampling (final stage spatial resolution)
        num_stages = len(self.depths)
        patches_res = self.backbone.patches_resolution
        self.grid_size = (
            patches_res[0] // (2 ** (num_stages - 1)),
            patches_res[1] // (2 ** (num_stages - 1)),
        )

        # Flatten all FocalNetBlocks across all BasicLayers into one list
        self.blocks = [block for layer in self.backbone.layers for block in layer.blocks]

        # cumulative block-end index per stage (1-based)
        depth_prefix_sum_list = []
        s = 0
        for d in self.depths:
            s += d
            depth_prefix_sum_list.append(s)

        # PatchEmbed downsamplers at end of each stage (None for last stage)
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
        self._H, self._W = H, W
        return x

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        """
        EoMT-aware FocalNet block runner.

        FocalNetBlock.forward() expects (B, H*W, C) and reads block.H / block.W
        internally to reshape. We set those before every call.

        x  : (B, L, C)  — flat spatial tokens
        q  : (num_q, C) — unbatched query tokens
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

            if q.dim() == 2:
                q_batched = q[None].expand(B, -1, -1)
            else:
                q_batched = q

            # 1) Concatenate query + image tokens: (B, num_q+L, C)
            xq = torch.cat([q_batched, x], dim=1)

            # 2) Pre-attention norm via block.norm1
            pre_attn = block.norm1(xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]   # (B, num_q, C)
            x_tok = pre_attn[:, eomt_obj.num_q :, :]   # (B, L, C)

            # 3) EoMT prediction heads
            mask_logits, class_logits = eomt_obj._predict(x_tok, q_tok)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)

            # 4) EoMT cross-attention
            after_eomt = torch.cat([q_tok, x_tok], dim=1)
            new_x = eomt_obj.attn[eomt_idx](self.norm(after_eomt))
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[eomt_idx](new_x))

            # 5) Split back
            q_batched = xq[:, : eomt_obj.num_q, :]
            x = xq[:, eomt_obj.num_q :, :]

            # 6) Standard FocalNet modulation block on image tokens
            x = block(x)

            # 7) Mirror FFN sub-layer for query tokens (no focal modulation)
            q_batched = q_batched + block.drop_path(
                block.gamma_2 * (
                    block.norm2(block.mlp(q_batched))
                    if block.use_postln
                    else block.mlp(block.norm2(q_batched))
                )
            )

            q = q_batched

        else:
            x = block(x)

        # Downsample at stage boundary via PatchEmbed
        ds_idx = str(i + 1)
        if ds_idx in self.patch_embed_dict:
            x_2d = x.transpose(1, 2).reshape(B, C, H, W)
            x, H, W = self.patch_embed_dict[ds_idx](x_2d)
            self._H, self._W = H, W

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        """Apply final backbone LayerNorm after all blocks."""
        x = self.norm(x)
        return x, q


def load_focalnet_ckpt(model: FocalNet, ckpt_path: str) -> FocalNet:
    """Load a FocalNet checkpoint, tolerating common key/shape mismatches."""
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

    # Strip backbone. prefix (e.g. from segmentation checkpoints)
    state_dict = {
        k[9:] if k.startswith("backbone.") else k: v
        for k, v in state_dict.items()
    }

    # Drop classification head — not used for segmentation
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("FocalNet — Missing keys :", missing)
    print("FocalNet — Unexpected keys:", unexpected)
    return model
