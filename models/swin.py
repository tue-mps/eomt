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

        sizes = swin_sizes[swin_size]
        self.num_heads = sizes["num_heads"]
        self.embed_dim = self.backbone.embed_dim * 2 ** (len(self.num_heads) - 1)
        self.patch_size = self.backbone.patch_embed.patch_size
        self.grid_size = tuple(
            x // (2 ** (len(self.num_heads) - 1))
            for x in self.backbone.patch_embed.patches_resolution
        )
        self.num_prefix_tokens = 0
        self.depths = sizes["depths"]

        depth_prefix_sum_list = [0]
        for d in self.depths:
            depth_prefix_sum_list.append(depth_prefix_sum_list[-1] + d)
        depth_prefix_sum_list.pop(0)

        # downsample (patch merging) modules keyed by the *ending* block index in that stage
        self.patch_merging_dict = nn.ModuleDict(
            {str(d): layer.downsample for d, layer in zip(depth_prefix_sum_list, self.backbone.layers)}
        )

        # flatten all Swin blocks into a single list for iteration
        self.blocks = [block for layer in self.backbone.layers for block in layer.blocks]
        self.norm = self.backbone.norm
        self.attn_multiplier = multiplier

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        # Swin keeps features in B,C,H,W; patch_embed already does that.
        x = self.backbone.patch_embed(x)
        return x

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        """
        EoMT-aware Swin block runner.

        Swin blocks operate on B,C,H,W internally but are implemented as transformers
        over sequences using windowed attention. Here we:
        - treat x as [B, N, C] tokens for EoMT,
        - then reshape back to B,C,H,W for the Swin block and downsample.
        """
        B, C, H, W = x.shape

        # Flatten to tokens for EoMT (channel-last) if we are in the EoMT region
        if i >= len(self.blocks) - eomt_obj.num_blocks:
            x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)

            # 1) concat q and x
            xq = torch.cat((q[None, :, :].expand(x_seq.shape[0], -1, -1), x_seq), dim=1)

            # 2) pre-attention norm using Swin block.norm1
            pre_attn = block.norm1(xq)
            x_tok, q_tok = pre_attn[:, eomt_obj.num_q :, :], pre_attn[:, : eomt_obj.num_q, :]

            # 3) EoMT prediction
            mask_logits, class_logits = eomt_obj._predict(x_tok, q_tok)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)

            # 4) EoMT cross-attention
            after_eomt = torch.cat((q_tok, x_tok), dim=1)
            new_x = eomt_obj.attn[i - len(self.blocks)](self.norm(after_eomt))
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[i - len(self.blocks)](new_x))

            # 5) split back into x and q
            x_seq, q = xq[:, eomt_obj.num_q :, :], xq[:, : eomt_obj.num_q, :]

            # 6) reshape x_seq back to B,C,H,W for the Swin block
            x = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)

            # 7) standard Swin block (attn + mlp with its own drop_path)
            x = block(x)

            # 8) q_mlp integrated here: apply the block's MLP normalization and drop_path to q.
            # Swin blocks do not expose ls2 / drop_path2, only a single drop_path.
            # We reuse norm2 + mlp + drop_path on q in token space.
            q = q + block.drop_path(block.mlp(block.norm2(q)))

        else:
            # plain Swin block without EoMT
            x = block(x)

        # handle downsampling between stages
        ds_idx = str(i + 1)
        if ds_idx in self.patch_merging_dict and self.patch_merging_dict[ds_idx] is not None:
            x = self.patch_merging_dict[ds_idx](x)

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        # final norm operates on sequence in original Swin, but here x is B,C,H,W.
        # flatten, norm, then reshape back.
        B, C, H, W = x.shape
        x_seq = x.permute(0, 2, 3, 1).reshape(B, H * W, C)
        x_seq = self.norm(x_seq)
        x = x_seq.reshape(B, H, W, C).permute(0, 3, 1, 2)
        return x, q


def load_swin_ckpt_ignore_attn_mask(model, ckpt_path):
    # 1) Load checkpoint state dict
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
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
