# based on https://github.com/10-zin/Synthesizer/blob/master/synth/synthesizer/modules.py
import math
from functools import partial

import torch
from timm.models import register_model
from torch import nn, einsum
from torch.nn import functional as F
from timm.models.vision_transformer import Block, PatchEmbed
from timm.models.vision_transformer import (
    VisionTransformer,
)
from models.backbones.utils import vit_sizes


class FactorizedRandomAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads,
        max_seq_len,
        f,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
    ):
        super().__init__()
        self.random_attn_1 = nn.Parameter(torch.randn(num_heads, max_seq_len, f))
        self.random_attn_2 = nn.Parameter(torch.randn(num_heads, f, max_seq_len))
        self.dropout = nn.Dropout(attn_drop)
        self.num_heads = num_heads
        self.v = nn.Linear(dim, dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        random_attn = self.random_attn_1 @ self.random_attn_2  # H x ms x ms
        random_attn = random_attn[:, :N, :N]  # H x N x N
        v = self.v(x).reshape(B, N, self.num_heads, C // self.num_heads).transpose(1, 2)  # B x H x N x d

        if attn_mask is not None:
            random_attn = random_attn.masked_fill(attn_mask == 0, -1e9)

        random_attn = self.dropout(F.softmax(random_attn, dim=-1))
        output = einsum("hnm,bhmd->bhnd", random_attn, v)
        output = output.transpose(1, 2).reshape(B, N, C)
        output = self.proj(output)
        output = self.proj_drop(output)

        return output


class FactorizedRandomBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        max_seq_len,
        f,
        qkv_bias=False,
        proj_drop=0.0,
        attn_drop=0.0,
        **kwargs,
    ):
        super().__init__(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            proj_drop=proj_drop,
            attn_drop=attn_drop,
            **kwargs,
        )
        self.attn = FactorizedRandomAttention(
            dim,
            num_heads,
            max_seq_len,
            f,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
        )


class FRSynthesizer(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        f=8,
        class_token=True,
        qkv_bias=True,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        init_values=None,
        no_embed_class=True,
        pre_norm=False,
        fc_norm=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        weight_init="",
        embed_layer=PatchEmbed,
        norm_layer=None,
        act_layer=None,
        **kwargs,
    ):
        self.patch_size = patch_size
        seq_len = (img_size / patch_size) ** 2 + (1 if class_token else 0)
        rest = math.ceil(seq_len / f)
        max_seq_len = f * rest
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            class_token=class_token,
            qkv_bias=qkv_bias,
            block_fn=partial(FactorizedRandomBlock, max_seq_len=max_seq_len, f=f),
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer
        )
        self.max_seq_len = max_seq_len
        self.f = f

    def set_image_res(self, res):
        new_seq_len = (res / self.patch_size) ** 2 + self.num_prefix_tokens
        rest = math.ceil(new_seq_len / self.f)
        new_max_seq_len = self.f * rest
        old_max_seq_len = self.max_seq_len
        print(f"Changing resolutions: max_seq_len: {old_max_seq_len} -> {new_max_seq_len}")
        self.max_seq_len = new_max_seq_len

        for block in self.blocks:
            r_1 = F.interpolate(
                block.attn.random_attn_1.transpose(1, 2),
                size=new_max_seq_len,
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
            r_2 = F.interpolate(
                block.attn.random_attn_2,
                size=new_max_seq_len,
                mode="linear",
                align_corners=False,
            )
            block.attn.random_attn_1 = nn.Parameter(r_1.contiguous())
            block.attn.random_attn_2 = nn.Parameter(r_2.contiguous())


@register_model
def synthesizer_fr_vit_tiny_patch16(pretrained=False, img_size=224, f=8, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = FRSynthesizer(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        f=f,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs,
    )
    return model




@register_model
def synthesizer_fr_vit_small_patch16(pretrained=False, img_size=224, f=8, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = FRSynthesizer(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        f=f,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs,
    )
    return model


@register_model
def synthesizer_fr_vit_base_patch16(pretrained=False, img_size=224, f=8, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = FRSynthesizer(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        f=f,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs,
    )
    return model



@register_model
def synthesizer_fr_vit_large_patch16(pretrained=False, img_size=224, f=8, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = FRSynthesizer(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        f=f,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs,
    )
    return model
