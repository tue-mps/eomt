import math
from functools import partial

import torch
import torch.nn.functional as F
from timm.models import register_model
from timm.models.vision_transformer import Block, PatchEmbed
from torch import nn

from timm.models.vision_transformer import (
    VisionTransformer,
)
from models.backbones.utils import vit_sizes



def default(val, default_val):
    return val if val is not None else default_val


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


class LinformerSelfAttention(nn.Module):
    # Taken from https://github.com/lucidrains/linformer with some modifications
    def __init__(
        self,
        dim,
        seq_len,
        k=256,
        heads=8,
        dim_head=None,
        one_kv_head=False,
        share_kv=False,
        dropout=0.0,
    ):
        super().__init__()
        assert (dim % heads) == 0, "dimension must be divisible by the number of heads"

        self.seq_len = seq_len
        self.k = k

        self.heads = heads

        dim_head = default(dim_head, dim // heads)
        self.dim_head = dim_head

        self.to_q = nn.Linear(dim, dim_head * heads, bias=False)

        kv_dim = dim_head if one_kv_head else (dim_head * heads)
        self.to_k = nn.Linear(dim, kv_dim, bias=False)
        self.proj_k = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.share_kv = share_kv
        if not share_kv:
            self.to_v = nn.Linear(dim, kv_dim, bias=False)
            self.proj_v = nn.Parameter(init_(torch.zeros(seq_len, k)))

        self.dropout = nn.Dropout(dropout)
        self.to_out = nn.Linear(dim_head * heads, dim)

    def resize(self, seq_len):
        if self.seq_len == seq_len:
            return

        # self.proj_k -> seq_len (old) x k
        old_imsize = int(math.sqrt(self.seq_len))
        new_imsize = int(math.sqrt(seq_len))
        print(f"resizing from {old_imsize} to {new_imsize}")
        extra_tokens = self.seq_len - old_imsize**2
        proj_k = self.proj_k.permute(1, 0).view(1, self.k, self.seq_len)  # 1 x k x seq_len (old)
        proj_k_tokens, proj_k = proj_k[:, :, :extra_tokens], proj_k[:, :, extra_tokens:].view(
            1, self.k, old_imsize, old_imsize
        )
        proj_k = F.interpolate(
            proj_k, size=(new_imsize, new_imsize), mode="bilinear"
        )  # 1 x k x imsize (new) x imsize (new)
        proj_k = torch.cat((proj_k_tokens, proj_k.view(1, self.k, -1)), dim=-1)
        proj_k = proj_k.view(self.k, -1).permute(1, 0)  # seq_len (new) x k
        self.proj_k = nn.Parameter(proj_k.contiguous())

        if not self.share_kv:
            proj_v = self.proj_v.permute(1, 0).view(1, self.k, self.seq_len)
            proj_v_tokens, proj_v = proj_v[:, :, :extra_tokens], proj_v[:, :, extra_tokens:].view(
                1, self.k, old_imsize, old_imsize
            )
            proj_v = F.interpolate(
                proj_v, size=(new_imsize, new_imsize), mode="bilinear"
            )  # 1 x k x imsize (new) x imsize (new)
            proj_v = torch.cat((proj_v_tokens, proj_v.view(1, self.k, -1)), dim=-1)
            proj_v = proj_v.view(self.k, -1).permute(1, 0)
            self.proj_v = nn.Parameter(proj_v.contiguous())

        self.seq_len = seq_len

    def forward(self, x, context=None, **kwargs):
        b, n, _, d_h, h, k = *x.shape, self.dim_head, self.heads, self.k

        kv_len = n if context is None else context.shape[1]
        assert (
            kv_len == self.seq_len
        ), f"the sequence length of the key / values is set to be {self.seq_len}, but is {kv_len}"

        queries = self.to_q(x)

        def proj_seq_len(args):
            return torch.einsum("bnd,nk->bkd", *args)

        kv_input = x if context is None else context

        keys = self.to_k(kv_input)
        values = self.to_v(kv_input) if not self.share_kv else keys

        kv_projs = (self.proj_k, self.proj_v if not self.share_kv else self.proj_k)

        # project keys and values along the sequence length dimension to k

        keys, values = map(proj_seq_len, zip((keys, values), kv_projs))

        # merge head into batch for queries and key / values

        queries = queries.reshape(b, n, h, -1).transpose(1, 2)

        def merge_key_values(t):
            return t.reshape(b, k, -1, d_h).transpose(1, 2).expand(-1, h, -1, -1)

        keys, values = map(merge_key_values, (keys, values))

        # attention

        dots = torch.einsum("bhnd,bhkd->bhnk", queries, keys) * (d_h**-0.5)
        attn = dots.softmax(dim=-1)
        attn = self.dropout(attn)
        out = torch.einsum("bhnk,bhkd->bhnd", attn, values)

        # split heads
        out = out.transpose(1, 2).reshape(b, n, -1)
        return self.to_out(out)


class LinformerBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        seq_len,
        k=256,
        mlp_ratio=4.0,
        drop=0.0,
        attn_drop=0.0,
        init_values=None,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__(
            dim,
            num_heads,
            mlp_ratio=mlp_ratio,
            proj_drop=drop,
            init_values=init_values,
            drop_path=drop_path,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.attn = LinformerSelfAttention(dim, seq_len, k=k, heads=num_heads, dropout=attn_drop)


class LinformerViT(VisionTransformer):
    def __init__(
        self,
        img_size=224,
        patch_size=16,
        k=256,
        in_chans=3,
        num_classes=1000,
        global_pool="token",
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        init_values=None,
        class_token=True,
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
        self.num_patches = (img_size // patch_size) ** 2
        self.num_prefix_tokens = 1 if class_token else 0
        super().__init__(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            num_classes=num_classes,
            global_pool=global_pool,
            embed_dim=embed_dim,
            depth=depth,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            init_values=init_values,
            class_token=class_token,
            no_embed_class=no_embed_class,
            pre_norm=pre_norm,
            fc_norm=fc_norm,
            drop_rate=drop_rate,
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate,
            weight_init=weight_init,
            embed_layer=embed_layer,
            norm_layer=norm_layer,
            act_layer=act_layer,
            block_fn=partial(LinformerBlock, seq_len=self.num_patches + self.num_prefix_tokens, k=k),
        )

    def set_image_res(self):

        for block in self.blocks:
            block.attn.resize(self.num_patches + self.num_prefix_tokens)


@register_model
def linformer_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["Ti"]
    model = LinformerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs,
    )
    return model


@register_model
def linformer_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["S"]
    model = LinformerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs,
    )
    return model


@register_model
def linformer_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["B"]
    model = LinformerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs,
    )
    return model


@register_model
def linformer_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    sizes = vit_sizes["L"]
    model = LinformerViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **sizes,
        **kwargs,
    )
    return model
