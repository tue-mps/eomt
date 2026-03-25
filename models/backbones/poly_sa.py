from functools import partial

import torch
from timm.models import register_model
from torch import nn, einsum
from timm.models.vision_transformer import Block
from timm.models.vision_transformer import (
    VisionTransformer,
)
from utils import vit_sizes


class PolySelfAttention(nn.Module):
    """
    Implementation of the Poly-SA attention mechanism from the paper
    [Poly-NL: Linear Complexity Non-local Layers with Polynomials](https://arxiv.org/abs/2107.02859)
    """

    def __init__(
        self,
        dim,
        seq_len,
        num_heads=8,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        phi=nn.Identity(),
        norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        assert dim % num_heads == 0, "dim should be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.phi = phi
        self.seq_len = seq_len
        self.p1 = nn.Parameter(torch.zeros(num_heads, seq_len))
        self.p2 = nn.Parameter(torch.zeros(num_heads, seq_len))
        self.norm = norm_layer(self.head_dim)

        nn.init.trunc_normal_(self.p1.data, std=0.02)
        nn.init.trunc_normal_(self.p2.data, std=0.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        )  # 3 x B x H x N x d
        q, k, v = qkv.unbind(0)  # B x H x N x d

        # collapse on p2
        modulator = einsum("hn,bhnd->bhd", self.p2, k * v)  # B x H x d
        modulator = self.attn_drop(self.norm(modulator))
        # expand again on p1
        modulator = einsum("hn,bhd->bhnd", self.p1, modulator)  # B x H x N x d

        x = (q * modulator).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class PolyBlock(Block):
    def __init__(
        self,
        dim,
        seq_len,
        num_heads,
        phi=nn.Identity(),
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        norm_layer=nn.LayerNorm,
        **kwargs
    ):
        super().__init__(
            dim, num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, norm_layer=norm_layer, proj_drop=proj_drop, **kwargs
        )
        self.attn = PolySelfAttention(
            dim,
            seq_len,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            phi=phi,
            norm_layer=norm_layer,
        )


class PolySAViT(VisionTransformer):
    def __init__(self, img_size=224, patch_size=16, class_token=True, **kwargs):
        self.seq_len = (img_size // patch_size) ** 2 + (1 if class_token else 0)
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            block_fn=partial(PolyBlock, seq_len=self.seq_len),
            class_token=class_token,
            **kwargs
        )

    def set_image_res(self, res):
        super().set_image_res(res)
        new_patched_size = res // self.patch_size
        num_extra_tokens = self.num_prefix_tokens
        new_seq_len = (res // self.patch_size) ** 2 + num_extra_tokens
        self.seq_len = new_seq_len
        for block in self.blocks:
            block.attn.seq_len = new_seq_len
            # p1, p2 are H x N
            p1 = block.attn.p1[:, num_extra_tokens:]
            H, _ = p1.shape

            # interpolate p1 in img dimensions
            p1_extra_tokens = block.attn.p1[:, :num_extra_tokens]
            orig_size = int(p1.shape[1] ** 0.5)
            new_p1 = nn.functional.interpolate(
                p1.reshape(1, H, orig_size, orig_size),
                size=(new_patched_size, new_patched_size),
                mode="bicubic",
                align_corners=False,
            )
            new_p1 = torch.cat((p1_extra_tokens, new_p1.reshape(H, -1)), dim=-1)
            assert new_p1.shape[0] == H and new_p1.shape[1] == new_seq_len
            block.attn.p1 = nn.Parameter(new_p1)

            # interpolate p2 in img dimensions
            p2 = block.attn.p2[:, num_extra_tokens:]
            p2_extra_tokens = block.attn.p2[:, :num_extra_tokens]
            new_p2 = nn.functional.interpolate(
                p2.reshape(1, H, orig_size, orig_size),
                size=(new_patched_size, new_patched_size),
                mode="bicubic",
                align_corners=False,
            )
            new_p2 = torch.cat((p2_extra_tokens, new_p2.reshape(H, -1)), dim=-1)
            assert new_p2.shape[0] == H and new_p2.shape[1] == new_seq_len
            block.attn.p2 = nn.Parameter(new_p2)


@register_model
def polysa_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = PolySAViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def polysa_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = PolySAViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def polysa_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = PolySAViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model


@register_model
def polysa_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = PolySAViT(
        img_size=img_size, patch_size=16, in_chans=3, norm_layer=partial(nn.LayerNorm, eps=1e-6), **sizes, **kwargs
    )
    return model
