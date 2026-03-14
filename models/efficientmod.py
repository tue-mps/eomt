from typing import Optional

import timm
import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

# Import the backbone module so its @register_model decorators run,
# registering efficient_mod_* names into timm's model registry.
import models.backbones.efficientmod  # noqa: F401
from models.backbones.efficientmod import EfficientMod


# Structural metadata per variant.
# depths / embed_dim / num_heads must match the backbone's @register_model configs.
efficientmod_sizes = {
    "XXS": {
        "backbone_name": "efficient_mod_xxs",
        "depths": [2, 2, 6, 2],
        "attention_depth": [0, 0, 1, 2],
        "embed_dim": [32, 64, 128, 256],
        "num_heads": [1, 2, 4, 8],  # per-stage, roughly embed_dim / 32
    },
    "XS": {
        "backbone_name": "efficient_mod_xs",
        "depths": [3, 3, 4, 2],
        "attention_depth": [0, 0, 3, 3],
        "embed_dim": [32, 64, 128, 256],
        "num_heads": [1, 2, 4, 8],
    },
    "S": {
        "backbone_name": "efficient_mod_s",
        "depths": [4, 4, 8, 4],
        "attention_depth": [0, 0, 4, 4],
        "embed_dim": [32, 64, 128, 256],
        "num_heads": [1, 2, 4, 8],
    },
}


class EfficientModModel(nn.Module):
    """
    EoMT-compatible wrapper for the EfficientMod backbone.

    EfficientMod is a hierarchical, channel-last model with 4 stages.
    Tokens are shaped (B, H, W, C) throughout the backbone, so this wrapper
    handles the BHWC <-> BLC reshaping needed by EoMT's flat-token protocol.
    """

    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "efficient_mod_s",
        efficientmod_size: str = "S",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        self.backbone: EfficientMod = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            num_classes=0,
            # EfficientMod backbone @register_model factories expect these kwargs
            layer_scale=False,
            layer_scale_init_values=1e-4,
            qkv_bias=True,
            drop_path_rate=0.1,
        )

        if ckpt_path is not None:
            self.backbone = load_efficientmod_ckpt(self.backbone, ckpt_path)

        sizes = efficientmod_sizes[efficientmod_size]
        self._stage_depths = sizes["depths"]
        self._attention_depths = sizes["attention_depth"]
        # depths = actual backbone block counts per stage (used by eomt.py for
        # get_layer_scale_list / get_attention_list sizing -- must match len(self.blocks))
        self.depths = self._stage_depths
        self.num_heads = sizes["num_heads"]
        self.attn_multiplier = multiplier

        # embed_dim at the final stage -- used by EoMT for query dimensionality
        self.embed_dim = sizes["embed_dim"][-1]
        self.num_prefix_tokens = 0

        # patch_size of the initial stem (for EoMT upscaler sizing)
        self.patch_size = (
            self.backbone.patch_embed.patch_size[0],
            self.backbone.patch_embed.patch_size[1],
        )

        # Final spatial resolution after all 4 downsampling stages:
        # stem stride=4, then 3x stride=2 => total stride = 32
        H_grid = img_size[0] // 32
        W_grid = img_size[1] // 32
        self.grid_size = (H_grid, W_grid)

        # Collect all blocks across all stages in one flat list,
        # matching block traversal order used in eomt.forward().
        # EfficientMod layers contain both BasicBlocks and AttentionBlocks.
        self.blocks = [blk for layer in self.backbone.layers for blk in layer.blocks]

        # Track per-stage (H, W) so run_block can maintain spatial dims
        # through the BHWC layout.
        self._stage_hw: list[tuple[int, int]] = []
        stride = 4
        h, w = img_size[0] // stride, img_size[1] // stride
        self._stage_hw.append((h, w))
        for _ in range(len(sizes["depths"]) - 1):
            h, w = h // 2, w // 2
            self._stage_hw.append((h, w))

        # Cumulative block counts at the end of each stage (1-based),
        # used to key the patch_merging_dict -- mirrors Swin's approach.
        depth_prefix_sum_list = []
        s = 0
        for d in self._stage_depths:
            s += d
            depth_prefix_sum_list.append(s)

        # Inter-stage downsamplers keyed by 1-based cumulative block index.
        # Only stages with an actual downsample module are registered.
        # EfficientMod's PatchMerging operates on BHWC tensors -- reshape
        # is handled in run_block before/after calling the downsampler.
        self.patch_merging_dict = nn.ModuleDict(
            {
                str(end): layer.downsample
                for end, layer in zip(depth_prefix_sum_list, self.backbone.layers)
                if layer.downsample is not None
            }
        )

        # Build a block-index -> (H, W) lookup table using actual stage depths
        # (not inflated depths + attention_depth).
        self._block_hw_map: list[tuple[int, int]] = []
        for stage_idx, stage_total in enumerate(self._stage_depths):
            for _ in range(stage_total):
                self._block_hw_map.append(self._stage_hw[stage_idx])

        self.norm = self.backbone.norm

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    # ------------------------------------------------------------------
    # EoMT interface
    # ------------------------------------------------------------------

    def pre_block(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply stem PatchEmbed, producing channel-last spatial tokens.

        EfficientMod expects (B, H, W, C) input to PatchEmbed.
        We store the spatial dims and return a flat (B, H*W, C) tensor
        so the EoMT loop can concatenate query tokens.
        """
        # backbone.patch_embed expects BHWC: permute BCHW -> BHWC
        x = self.backbone.patch_embed(x.permute(0, 2, 3, 1))
        B, H, W, C = x.shape
        self._H, self._W = H, W
        # Flatten to BLC for EoMT's standard token format
        return x.reshape(B, H * W, C).contiguous()

    def run_block(
        self,
        block: nn.Module,
        eomt_obj,
        x: torch.Tensor,
        q: torch.Tensor,
        i: int,
    ):
        """
        EoMT-aware EfficientMod block runner.

        x : (B, L, C)  -- flat spatial tokens (BLC layout)
        q : (num_q, C) or (B, num_q, C) -- query tokens
        i : global block index (0-based across all stages)

        EfficientMod BasicBlocks and AttentionBlocks both expect BHWC.
        AttentionBlocks additionally reshape to BLC internally, so
        they are compatible with either layout via their own reshape logic.
        """
        H, W = self._block_hw_map[i]
        B, L, C = x.shape
        assert L == H * W, f"Token count {L} != {H}x{W}={H * W}"

        total_blocks = len(self.blocks)

        if i >= total_blocks - eomt_obj.num_blocks:
            eomt_idx = i - (total_blocks - eomt_obj.num_blocks)

            if q.dim() == 2:
                q_batched = q[None].expand(B, -1, -1)
            else:
                q_batched = q

            # 1) Concatenate query + image tokens: (B, num_q + L, C)
            xq = torch.cat([q_batched, x], dim=1)

            # 2) Pre-attention norm -- EfficientMod blocks use block.norm (LayerNorm)
            pre_attn = block.norm(xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]   # (B, num_q, C)
            x_tok = pre_attn[:, eomt_obj.num_q :, :]   # (B, L, C)

            # 3) EoMT intermediate prediction
            mask_logits, class_logits = eomt_obj._predict(x_tok, q_tok)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)

            # 4) EoMT cross-attention on normed tokens
            after_eomt = torch.cat([q_tok, x_tok], dim=1)
            new_x = eomt_obj.attn[eomt_idx](self.norm(after_eomt))
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[eomt_idx](new_x))

            # 5) Split back
            q_batched = xq[:, : eomt_obj.num_q, :]
            x = xq[:, eomt_obj.num_q :, :]

            # 6) Run standard EfficientMod block on image tokens (BHWC)
            x_bhwc = x.reshape(B, H, W, x.shape[-1])
            x_bhwc = block(x_bhwc)
            x = x_bhwc.reshape(B, H * W, -1).contiguous()

            # 7) Mirror MLP sub-layer for query tokens
            #    BasicBlock: shortcut + drop_path(gamma * mlp(norm(x)))
            #    AttentionBlock: shortcut + drop_path(ls2(mlp(norm2(x))))
            if hasattr(block, "mlp"):
                if hasattr(block, "drop_path2"):
                    # AttentionBlock-style (timm Block subclass)
                    q_batched = q_batched + block.drop_path2(
                        block.ls2(block.mlp(block.norm2(q_batched)))
                    )
                else:
                    # BasicBlock-style
                    q_batched = q_batched + block.drop_path(
                        block.gamma_1 * block.mlp(block.norm(q_batched))
                    )

            q = q_batched

        else:
            # Standard forward: reshape BLC -> BHWC, run block, reshape back
            x_bhwc = x.reshape(B, H, W, C)
            x_bhwc = block(x_bhwc)
            x = x_bhwc.reshape(B, H * W, x_bhwc.shape[-1]).contiguous()

        # Update tracked spatial dims for post_blocks / _predict reshape
        self._H, self._W = H, W

        # Inter-stage downsampling -- mirrors Swin's patch_merging_dict pattern.
        # Key is the 1-based cumulative block count at the end of each stage.
        # EfficientMod's PatchMerging expects BHWC, so reshape before/after.
        ds_idx = str(i + 1)
        if ds_idx in self.patch_merging_dict:
            B2, L2, C2 = x.shape
            x_bhwc = x.reshape(B2, H, W, C2)
            x_bhwc = self.patch_merging_dict[ds_idx](x_bhwc)
            nB, nH, nW, nC = x_bhwc.shape
            x = x_bhwc.reshape(nB, nH * nW, nC).contiguous()
            self._H, self._W = nH, nW

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        """Apply final LayerNorm after all blocks (last-stage norm)."""
        B, L, C = x.shape
        x_bhwc = x.reshape(B, self._H, self._W, C)
        x_bhwc = self.norm(x_bhwc)
        # Return as (B, C, H, W) for EoMT _predict's upscaler path
        x_bchw = x_bhwc.permute(0, 3, 1, 2).contiguous()
        return x_bchw, q


def load_efficientmod_ckpt(model: EfficientMod, ckpt_path: str) -> EfficientMod:
    """Load an EfficientMod checkpoint, tolerating common key/shape mismatches."""
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

    # Drop classification head -- not used for segmentation
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("EfficientMod -- Missing keys :", missing)
    print("EfficientMod -- Unexpected keys:", unexpected)
    return model
