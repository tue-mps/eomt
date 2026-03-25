from typing import Optional

import torch
import torch.nn as nn
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from models.backbones.coat import CoaT  # noqa: F401


coat_sizes = {
    "coat_tiny":        {"embed_dims": [152, 152, 152, 152], "serial_depths": [2, 2, 2, 2], "parallel_depth": 6,  "num_heads": 8, "mlp_ratios": [4, 4, 4, 4]},
    "coat_mini":        {"embed_dims": [152, 216, 216, 216], "serial_depths": [2, 2, 2, 2], "parallel_depth": 6,  "num_heads": 8, "mlp_ratios": [4, 4, 4, 4]},
    "coat_small":       {"embed_dims": [152, 320, 320, 320], "serial_depths": [2, 2, 2, 2], "parallel_depth": 6,  "num_heads": 8, "mlp_ratios": [4, 4, 4, 4]},
    "coat_lite_tiny":   {"embed_dims": [64, 128, 256, 320],  "serial_depths": [2, 2, 2, 2], "parallel_depth": 0,  "num_heads": 8, "mlp_ratios": [8, 8, 4, 4]},
    "coat_lite_mini":   {"embed_dims": [64, 128, 320, 512],  "serial_depths": [2, 2, 2, 2], "parallel_depth": 0,  "num_heads": 8, "mlp_ratios": [8, 8, 4, 4]},
    "coat_lite_small":  {"embed_dims": [64, 128, 320, 512],  "serial_depths": [3, 4, 6, 3], "parallel_depth": 0,  "num_heads": 8, "mlp_ratios": [8, 8, 4, 4]},
    "coat_lite_medium": {"embed_dims": [128, 256, 320, 512], "serial_depths": [3, 6, 10, 8], "parallel_depth": 0, "num_heads": 8, "mlp_ratios": [4, 4, 4, 4]},
}


class CoaTSeg(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name: str = "coat_lite_small",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()

        sizes = coat_sizes[backbone_name]
        embed_dims     = sizes["embed_dims"]
        serial_depths  = sizes["serial_depths"]
        parallel_depth = sizes["parallel_depth"]
        num_heads      = sizes["num_heads"]
        mlp_ratios     = sizes["mlp_ratios"]

        self.backbone = CoaT(
            patch_size=4,
            in_chans=3,
            embed_dims=embed_dims,
            serial_depths=serial_depths,
            parallel_depth=parallel_depth,
            num_heads=num_heads,
            mlp_ratios=mlp_ratios,
            return_interm_layers=False,
        )

        if ckpt_path is not None:
            self.backbone = load_coat_ckpt(self.backbone, ckpt_path)

        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        num_stages = len(serial_depths) + (1 if parallel_depth > 0 else 0)
        self.num_heads = [num_heads] * num_stages

        self.embed_dim = embed_dims[-1]
        self.patch_size = (4, 4)

        self.depths = list(serial_depths) + ([parallel_depth] if parallel_depth > 0 else [])

        H, W = img_size
        self.grid_size = (
            H // (4 * 2 ** (len(serial_depths) - 1)),
            W // (4 * 2 ** (len(serial_depths) - 1)),
        )

        self._serial_blocks = [
            blk
            for stage in [
                self.backbone.serial_blocks1,
                self.backbone.serial_blocks2,
                self.backbone.serial_blocks3,
                self.backbone.serial_blocks4,
            ]
            for blk in stage
        ]
        self._parallel_blocks = list(self.backbone.parallel_blocks) if parallel_depth > 0 else []
        self.blocks = self._serial_blocks + self._parallel_blocks

        self._stage_ends = []
        s = 0
        for d in serial_depths:
            s += d
            self._stage_ends.append(s)

        self.norm = self.backbone.norm4

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std  = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        x1, (H1, W1) = self.backbone.patch_embed1(x)
        x1 = self.backbone.insert_cls(x1, self.backbone.cls_token1)
        self._sizes = [(H1, W1), None, None, None]
        self._B = x.shape[0]
        self._par_x2 = self._par_x3 = self._par_x4 = None
        return x1

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        B = x.shape[0]
        num_serial = len(self._serial_blocks)

        if i >= num_serial:
            par_idx = i - num_serial
            x1, x2, x3, x4 = self._parallel_blocks[par_idx](
                x, self._par_x2, self._par_x3, self._par_x4,
                sizes=self._sizes,
            )
            self._par_x2, self._par_x3, self._par_x4 = x2, x3, x4
            x = x4
            return x, q

        stage_idx = sum(i >= end for end in self._stage_ends)
        stage_idx = min(stage_idx, 3)
        H, W = self._sizes[stage_idx]

        eomt_window_start = num_serial - eomt_obj.num_blocks

        if i >= eomt_window_start:
            eomt_idx = i - eomt_window_start

            if q.dim() == 2:
                q_batched = q[None].expand(B, -1, -1)
            else:
                q_batched = q

            cls_tok = x[:, :1, :]
            img_tok = x[:, 1:, :]

            xq = torch.cat([q_batched, img_tok], dim=1)
            pre_attn = block.norm1(xq)
            q_tok = pre_attn[:, : eomt_obj.num_q, :]
            x_tok = pre_attn[:, eomt_obj.num_q :, :]

            mask_logits, class_logits = eomt_obj._predict(x_tok, q_tok)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)

            after_eomt = torch.cat([q_tok, x_tok], dim=1)
            new_x = eomt_obj.attn[eomt_idx](self.norm(after_eomt))
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[eomt_idx](new_x))

            q_batched = xq[:, : eomt_obj.num_q, :]
            img_tok   = xq[:, eomt_obj.num_q :, :]
            x = torch.cat([cls_tok, img_tok], dim=1)

            x = block(x, size=(H, W))
            q_batched = q_batched + block.drop_path(block.mlp(block.norm2(q_batched)))
            q = q_batched

        else:
            x = block(x, size=(H, W))

        idx_1based = i + 1
        next_embeds = [
            (self._stage_ends[0], self.backbone.patch_embed2, self.backbone.cls_token2, 1),
            (self._stage_ends[1], self.backbone.patch_embed3, self.backbone.cls_token3, 2),
            (self._stage_ends[2], self.backbone.patch_embed4, self.backbone.cls_token4, 3),
        ]
        for stage_end, pembed, cls_tok_param, next_idx in next_embeds:
            if idx_1based == stage_end:
                x_nocls = self.backbone.remove_cls(x)
                x_nocls = x_nocls.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
                x_new, (Hn, Wn) = pembed(x_nocls)
                x = self.backbone.insert_cls(x_new, cls_tok_param)
                self._sizes[next_idx] = (Hn, Wn)
                cache = [self._par_x2, self._par_x3, self._par_x4]
                cache[next_idx - 1] = x
                self._par_x2, self._par_x3, self._par_x4 = cache
                break

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        x = self.backbone.remove_cls(x)
        x = self.norm(x)
        return x, q


def load_coat_ckpt(model: CoaT, ckpt_path: str) -> CoaT:
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

    state_dict = {
        k[9:] if k.startswith("backbone.") else k: v
        for k, v in state_dict.items()
    }
    state_dict = {k: v for k, v in state_dict.items() if not k.startswith("head.")}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    print("CoaT — Missing keys :", missing)
    print("CoaT — Unexpected keys:", unexpected)
    return model
