from typing import Optional
import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn

from models.backbones.convViT import to_doubletuple


class BackboneNorm(nn.Module):
    def __init__(self, norm_layer: nn.Module, grid_size: Optional[tuple] = None):
        super().__init__()
        self.norm = norm_layer
        self.grid_size = grid_size  # (H, W)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Case 1: Spatial (B, C, H, W)
        if x.dim() == 4:
            return self.norm(x)
        # Case 2: Tokens (B, N, C)
        if x.dim() == 3:
            B, N, C = x.shape
            x_perm = x.transpose(1, 2).unsqueeze(-1)          # (B, C, N, 1)
            x_norm = self.norm(x_perm)                        # BN2d
            return x_norm.squeeze(-1).transpose(1, 2)         # (B, N, C)
        return x


def get_efficientvit_final_grid(input_resolution: int):
    final_size = (input_resolution + 31) // 32
    return (final_size, final_size)


class EfficientNet(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="efficient_vit_b2",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            num_classes=0,
        )
        # load ckpt and unwrap to .backbone (EfficientViT wrapper)
        self.backbone = load_swin_ckpt_ignore_attn_mask(self.backbone, ckpt_path).backbone

        self.embed_dim = self.backbone.width_list[-1]
        self.patch_size = to_doubletuple(32)
        self.grid_size = get_efficientvit_final_grid(img_size[0])
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier

        self.blocks = nn.ModuleList()
        self.depths = []
        self.num_heads = []
        self.patch_embed_dict = nn.ModuleDict()
        self.patch_embed_dict["0"] = self.backbone.input_stem

        # flatten stages into self.blocks; EfficientViT is CNN‑like, no explicit heads
        for stage_idx, stage in enumerate(self.backbone.stages):
            stage_depth = len(stage)
            self.depths.append(stage_depth)
            stage_heads = 8 if stage_idx >= 2 else 1  # dummy num_heads
            for block in stage:
                self.blocks.append(block)
                self.num_heads.append(stage_heads)

        if hasattr(self.backbone, "norm") and self.backbone.norm is not None:
            raw_norm = self.backbone.norm
        else:
            raw_norm = nn.BatchNorm2d(self.embed_dim, eps=1e-5)

        self.norm = BackboneNorm(raw_norm, grid_size=self.grid_size)

        self.B, self.C, self.H, self.W = 0, 0, 0, 0
        self.attn_multiplier = multiplier

        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        # stem only; stages are in self.blocks
        if "0" in self.patch_embed_dict:
            x = self.patch_embed_dict["0"](x)
            self.B, self.C, self.H, self.W = x.shape
        return x

    def q_mlp(self, block, q: torch.Tensor):
        """
        Apply the MLP/FFN part of an EfficientViT/EfficientNet MBConv‑style block to q
        (B, N, C) in token space, skipping spatial depthwise conv. This is the
        "q_mlp" action, now used from run_block.
        """
        target_module = block
        use_residual = False

        # ResidualBlock(main, shortcut) style
        if hasattr(block, "main") and hasattr(block, "shortcut"):
            is_identity = isinstance(block.shortcut, (nn.Identity, type(None))) or (
                isinstance(block.shortcut, nn.Sequential) and len(block.shortcut) == 0
            )
            if is_identity:
                use_residual = True
            target_module = block.main

        # EfficientViTBlock: local_module holds MBConv
        if hasattr(target_module, "local_module"):
            return self.q_mlp(target_module.local_module, q)

        # MBConv: channel‑mixing via inverted_conv + point_conv
        if hasattr(target_module, "inverted_conv") and hasattr(target_module, "point_conv"):
            B, N, C = q.shape
            q_reshaped = q.transpose(1, 2).unsqueeze(-1)  # (B, C, N, 1)

            x = target_module.inverted_conv(q_reshaped)   # expansion 1x1 + BN + act
            # depthwise conv is skipped for queries
            x = target_module.point_conv(x)               # projection 1x1 + BN

            q_out = x.squeeze(-1).transpose(1, 2)         # (B, N, C)

            if use_residual:
                q_out = q_out + q
            return q_out

        # fallback: unknown structure
        return q

    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i: int):
        """
        Unified EoMT run_block for EfficientNet/EfficientViT.

        - If x is spatial (B,C,H,W), we apply the backbone block in spatial domain.
        - If we are in the EoMT region, we:
          * flatten x to tokens,
          * run EoMT (predict + cross‑attend),
          * apply q_mlp via MBConv logic,
          * reshape back to spatial and run the block.
        """
        # Spatial path
        if x.dim() == 4:
            B, C, H, W = x.shape
            self.B, self.C, self.H, self.W = B, C, H, W

            if i >= len(self.blocks) - eomt_obj.num_blocks:
                # flatten to tokens (B, N, C)
                x_tokens = x.flatten(2).transpose(1, 2)

                # EoMT interaction
                xq = torch.cat((q[None, :, :].expand(x_tokens.shape[0], -1, -1), x_tokens), dim=1)
                pre_attn = self.norm(xq)
                x_tok, q_tok = pre_attn[:, eomt_obj.num_q :, :], pre_attn[:, : eomt_obj.num_q, :]

                mask_logits, class_logits = eomt_obj._predict(x_tok, q_tok)
                eomt_obj.mask_logits_per_layer.append(mask_logits)
                eomt_obj.class_logits_per_layer.append(class_logits)

                after_eomt = torch.cat((q_tok, x_tok), dim=1)
                new_x = eomt_obj.attn[i - len(self.blocks)](self.norm(after_eomt))
                xq = xq + eomt_obj.dp(eomt_obj.ls_list[i - len(self.blocks)](new_x))
                x_tokens, q = xq[:, eomt_obj.num_q :, :], xq[:, : eomt_obj.num_q, :]

                # q_mlp integrated here
                q = self.q_mlp(block, q)

                # reshape x back to spatial, run block normally
                x = x_tokens.transpose(1, 2).reshape(B, C, H, W)
                x = block(x)
            else:
                x = block(x)

            # update cache/grid
            self.B, self.C, self.H, self.W = x.shape
            self.grid_size = (self.H, self.W)
            return x, q

        # Token path (if upstream passes tokens directly)
        elif x.dim() == 3:
            B, N, C = x.shape
            H, W = self.grid_size
            if H * W != N and self.H * self.W == N:
                H, W = self.H, self.W

            if i >= len(self.blocks) - eomt_obj.num_blocks:
                xq = torch.cat((q[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)
                pre_attn = self.norm(xq)
                x_tok, q_tok = pre_attn[:, eomt_obj.num_q :, :], pre_attn[:, : eomt_obj.num_q, :]

                mask_logits, class_logits = eomt_obj._predict(x_tok, q_tok)
                eomt_obj.mask_logits_per_layer.append(mask_logits)
                eomt_obj.class_logits_per_layer.append(class_logits)

                after_eomt = torch.cat((q_tok, x_tok), dim=1)
                new_x = eomt_obj.attn[i - len(self.blocks)](self.norm(after_eomt))
                xq = xq + eomt_obj.dp(eomt_obj.ls_list[i - len(self.blocks)](new_x))
                x, q = xq[:, eomt_obj.num_q :, :], xq[:, : eomt_obj.num_q, :]

                # q_mlp here as well
                q = self.q_mlp(block, q)

            # always run block in spatial domain
            x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
            x_spatial = block(x_spatial)
            self.B, self.C, self.H, self.W = x_spatial.shape
            self.grid_size = (self.H, self.W)
            x_tokens = x_spatial.flatten(2).transpose(1, 2)
            return x_tokens, q

        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        # consistent with other backbones: final norm on tokens if needed
        if x.dim() == 4:
            B, C, H, W = x.shape
            x_tokens = x.flatten(2).transpose(1, 2)
            x_tokens = self.norm(x_tokens)
            x = x_tokens.transpose(1, 2).reshape(B, C, H, W)
        else:
            x = self.norm(x)
        return x, q


def load_swin_ckpt_ignore_attn_mask(model, ckpt_path):
    if ckpt_path is None:
        return model
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "model_state" in checkpoint:
        state_dict = {
            k[7:] if k.startswith("module.") else k: v
            for k, v in checkpoint["model_state"].items()
        }
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
    else:
        state_dict = model.state_dict()

    model_state = model.state_dict()
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model
