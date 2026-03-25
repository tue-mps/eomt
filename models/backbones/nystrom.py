# Taken from https://github.com/mlpen/Nystromformer/blob/main/ImageNet/T2T-ViT/models/token_nystromformer.py with slight modifications

from functools import partial

import torch
import torch.nn as nn
from timm.models import register_model
from timm.models.vision_transformer import Block
from timm.models.vision_transformer import (
    VisionTransformer,
)
from utils import vit_sizes


class NysAttention(nn.Module):
    def __init__(
        self,
        dim,
        num_heads=8,
        qkv_bias=False,
        qk_scale=None,
        attn_drop=0.0,
        proj_drop=0.0,
        num_landmarks=64,
        kernel_size=0,
        init_option="exact",
    ):
        super().__init__()
        self.num_heads = num_heads
        self.dim = dim
        self.head_dim = dim // num_heads
        self.scale = self.head_dim**0.5
        self.landmarks = num_landmarks
        self.kernel_size = kernel_size
        self.init_option = init_option

        self.qkv = nn.Linear(dim, self.head_dim * 3 * num_heads, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        if self.kernel_size > 0:
            self.conv = nn.Conv2d(
                in_channels=self.num_heads,
                out_channels=self.num_heads,
                kernel_size=(self.kernel_size, 1),
                padding=(self.kernel_size // 2, 0),
                bias=False,
                groups=self.num_heads,
            )

    def iterative_inv(self, mat, n_iter=6):
        Id_mat = torch.eye(mat.size(-1), device=mat.device)
        K = mat

        # The entries of K are positive and ||K||_{\infty} = 1 due to softmax
        if self.init_option == "original":
            # This original implementation is more conservative to compute coefficient of Z_0.
            V = 1 / torch.max(torch.sum(K, dim=-2)) * K.transpose(-1, -2)
        else:
            # This is the exact coefficient computation, 1 / ||K||_1, of initialization of Z_0, leading to faster convergence.
            V = 1 / torch.max(torch.sum(K, dim=-2), dim=-1).values[:, :, None, None] * K.transpose(-1, -2)

        for _ in range(n_iter):
            KV = torch.matmul(K, V)
            V = torch.matmul(
                0.25 * V,
                13 * Id_mat - torch.matmul(KV, 15 * Id_mat - torch.matmul(KV, 7 * Id_mat - KV)),
            )
        return V

    def forward(self, x, attn_mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        q /= self.scale

        keys_head_dim = k.size(-1)
        segs = N // self.landmarks
        if N % self.landmarks == 0:
            keys_landmarks = k.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(
                dim=-2
            )
            queries_landmarks = q.reshape(B, self.num_heads, self.landmarks, N // self.landmarks, keys_head_dim).mean(
                dim=-2
            )
        else:
            num_k = (segs + 1) * self.landmarks - N
            assert segs > 0 and num_k > 0, f"segs or num_k is zero: segs={segs}, num_k={num_k}"
            keys_landmarks_f = (
                k[:, :, : num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim=-2)
            )
            keys_landmarks_l = (
                k[:, :, num_k * segs :, :]
                .reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim)
                .mean(dim=-2)
            )
            keys_landmarks = torch.cat((keys_landmarks_f, keys_landmarks_l), dim=-2)

            queries_landmarks_f = (
                q[:, :, : num_k * segs, :].reshape(B, self.num_heads, num_k, segs, keys_head_dim).mean(dim=-2)
            )
            queries_landmarks_l = (
                q[:, :, num_k * segs :, :]
                .reshape(B, self.num_heads, self.landmarks - num_k, segs + 1, keys_head_dim)
                .mean(dim=-2)
            )
            queries_landmarks = torch.cat((queries_landmarks_f, queries_landmarks_l), dim=-2)

        kernel_1 = torch.nn.functional.softmax(torch.matmul(q, keys_landmarks.transpose(-1, -2)), dim=-1)
        kernel_2 = torch.nn.functional.softmax(
            torch.matmul(queries_landmarks, keys_landmarks.transpose(-1, -2)), dim=-1
        )
        kernel_3 = torch.nn.functional.softmax(torch.matmul(queries_landmarks, k.transpose(-1, -2)), dim=-1)

        x = torch.matmul(
            torch.matmul(kernel_1, self.iterative_inv(kernel_2)),
            torch.matmul(kernel_3, v),
        )

        if self.kernel_size > 0:
            x += self.conv(v)

        x = x.transpose(1, 2).reshape(B, N, self.dim)
        x = self.proj(x)
        x = self.proj_drop(x)
        return v.transpose(1, 2).reshape(B, N, self.dim) + x


class NystromBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        num_landmarks=64,
        kernel_size=0,
        init_option="exact",
        proj_drop=0.0,
        **kwargs,
    ):
        super().__init__(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            **kwargs,
        )
        self.attn = NysAttention(
            dim,
            num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=proj_drop,
            num_landmarks=num_landmarks,
            kernel_size=kernel_size,
            init_option=init_option,
        )


class NystromViT(VisionTransformer):
    def __init__(self, *args, num_landmarks=64, init_option="exact", kernel_size=0, **kwargs):
        super().__init__(
            *args,
            block_fn=partial(
                NystromBlock,
                num_landmarks=num_landmarks,
                init_option=init_option,
                kernel_size=kernel_size,
            ),
            **kwargs,
        )

@register_model
def nystrom64_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["Ti"]
    model = NystromViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_landmarks=64,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def nystrom32_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["Ti"]
    model = NystromViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_landmarks=32,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def nystrom64_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["S"]
    model = NystromViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_landmarks=64,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def nystrom32_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["S"]
    model = NystromViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_landmarks=32,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def nystrom64_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["B"]
    model = NystromViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_landmarks=64,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def nystrom32_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["B"]
    model = NystromViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_landmarks=32,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def nystrom64_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["L"]
    model = NystromViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_landmarks=64,
        **sizes,
        **kwargs,
    )
    return model


@register_model
def nystrom32_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    sizes = vit_sizes["L"]
    model = NystromViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        num_landmarks=32,
        **sizes,
        **kwargs,
    )
    return model