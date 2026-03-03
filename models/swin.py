from typing import Optional
from collections import OrderedDict

import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn

from models.backbones.swin import SwinTransformer, swin_sizes


class Swin(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "swin_large_patch4_window7_224",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
        swin_size: str = "Ti",
    ):
        super().__init__()

        self.backbone: SwinTransformer = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            num_classes=0,
        )
        self.backbone.set_image_res(img_size)
        self.backbone = load_swin_ckpt_ignore_attn_mask(self.backbone, ckpt_path)

        # num_heads from swin_sizes — used by EoMT to build its attn/ls modules
        sizes = swin_sizes[swin_size]
        self.num_heads = sizes["num_heads"]
        self.attn_multiplier = multiplier

        # all structural properties derived from the actual instantiated backbone
        self.embed_dim = self.backbone.num_features  # already = embed_dim * 2^(num_layers-1)
        self.patch_size = self.backbone.patch_embed.patch_size
        self.grid_size = tuple(
            x // (2 ** (len(sizes["depths"]) - 1))
            for x in self.backbone.patch_embed.patches_resolution
        )
        self.num_prefix_tokens = 0

        # depths from the live backbone — never from swin_sizes
        # swin_sizes may have fewer stages than the timm model (e.g. 3 vs 4)
        self.depths = sizes["depths"]

        # cumulative block-end index per stage (1-based)
        depth_prefix_sum_list = []
        s = 0
        for d in self.depths:
            s += d
            depth_prefix_sum_list.append(s)

        # PatchMerging modules keyed by cumulative block count at each stage end.
        # Only stages with an actual downsample (all but the last) are registered.
        self.patch_merging_dict = nn.ModuleDict(
            {
                str(end): layer.downsample
                for end, layer in zip(depth_prefix_sum_list, self.backbone.layers)
                if layer.downsample is not None
            }
        )

        # flatten all Swin blocks into a single list for iteration
        self.blocks = [block for layer in self.backbone.layers for block in layer.blocks]
        self.norm = self.backbone.norm

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        # patch_embed outputs (B, H*W, C) — native token layout of backbones/swin.py
        x = self.backbone.patch_embed(x)
        if self.backbone.ape:
            x = x + self.backbone.absolute_pos_embed
        x = self.backbone.pos_drop(x)
        return x

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        """
        EoMT-aware Swin block runner.

        backbones/swin.py SwinTransformerBlock.forward() takes x of shape
        (B, H*W, C). All window-partition, cyclic-shift, W-MSA/SW-MSA, and
        FFN logic lives inside the block and operates on that flat layout.

        x stays as (B, L, C) throughout.
        q stays as (num_q, C) — unbatched — consistent with EoMT.forward().
        """
        H, W = block.input_resolution
        B, L, C = x.shape
        assert L == H * W, (
            f"Token count {L} does not match block resolution {H}x{W}={H * W}"
        )

        if i >= len(self.blocks) - eomt_obj.num_blocks:
            eomt_idx = i - (len(self.blocks) - eomt_obj.num_blocks)

            if q.dim() == 2:
                q_batched = q[None].expand(B, -1, -1)
            else:
                q_batched = q

            # 1) concat query tokens in front of image tokens: (B, num_q+L, C)
            xq = torch.cat([q_batched, x], dim=1)

            # 2) pre-attention norm via block.norm1
            #    correctly sized for this stage's C (not the final-stage norm)
            pre_attn = block.norm1(xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]  # (B, num_q, C)
            x_tok = pre_attn[:, eomt_obj.num_q :, :]  # (B, L, C)

            # 3) EoMT prediction heads
            mask_logits, class_logits = eomt_obj._predict(x_tok, q_tok)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)

            # 4) EoMT cross-attention
            #    CrossAttentionSingleInput carries its own norm — do NOT apply
            #    self.norm here (it is LayerNorm sized for the final stage only)
            after_eomt = torch.cat([q_tok, x_tok], dim=1)  # (B, num_q+L, C)
            new_x = eomt_obj.attn[eomt_idx](self.norm(after_eomt))
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[eomt_idx](new_x))

            # 5) split back
            q_batched = xq[:, : eomt_obj.num_q, :]  # (B, num_q, C)
            x = xq[:, eomt_obj.num_q :, :]          # (B, L, C)

            # 6) standard Swin block: (B, L, C) -> (B, L, C)
            #    handles norm1, cyclic-shift, window-partition,
            #    W-MSA/SW-MSA, reverse-shift, residual, norm2, MLP, residual
            x = block(x)

            # 7) mirror Swin FFN sub-layer for query tokens:
            #    shortcut + drop_path(mlp(norm2(q)))
            q_batched = q_batched + block.drop_path(block.mlp(block.norm2(q_batched)))

            # collapse batch dim back to (num_q, C) as expected by EoMT
            q = q_batched

        else:
            # plain Swin block — (B, L, C) in, (B, L, C) out
            x = block(x)

        # PatchMerging between stages: (B, H*W, C) -> (B, H/2*W/2, 2C)
        # key is the 1-based cumulative block count at the end of each stage
        ds_idx = str(i + 1)
        if ds_idx in self.patch_merging_dict:
            x = self.patch_merging_dict[ds_idx](x)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        # backbone norm operates directly on (B, L, C) — no reshape needed
        x = self.norm(x)
        return x, q


def load_swin_ckpt_ignore_attn_mask(model, ckpt_path):
    if ckpt_path is None:
        return model

    # 1) Load checkpoint state dict
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    state_dict = None
    if "model_state" in checkpoint:
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in checkpoint["model_state"].items()
        }
    elif "teacher" in checkpoint:
        state_dict = {
            k[9:] if k.startswith("backbone.") else k: v
            for k, v in checkpoint["teacher"].items()
        }
    else:
        state_dict = checkpoint

    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]

    new_state_dict = OrderedDict()
    # 2) Build a reference of current model shapes for conditional filtering
    model_state = model.state_dict()
    for k, v in state_dict.items():
        if "attn_mask" in k and k in model_state:
            if v.shape != model_state[k].shape:
                print(
                    f"Skipping {k}: ckpt {tuple(v.shape)} != model {tuple(model_state[k].shape)}"
                )
            # skip mismatched attn_mask (recomputed at runtime)
            continue
        new_state_dict[k] = v

    # 3) Load with strict=False to tolerate any remaining non-critical diffs
    missing, unexpected = model.load_state_dict(new_state_dict, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model
