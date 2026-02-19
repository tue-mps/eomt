# Taken from https://github.com/robflynnyh/hydra-linear-attention/blob/main/hydra.py with slight changes
from timm.models import register_model
from timm.models.vision_transformer import Block,PatchEmbed
from torch import nn
from timm.models.vision_transformer import (
    VisionTransformer,
)


from models.backbones.utils import vit_sizes

class HydraAttention(nn.Module):
    """Attention from HydraViT."""

    def __init__(self, dim, output_layer="linear", drop=0.0, proj_drop=None, qkv_bias=True):
        """Create Hydra attention layer.

        Args:
            dim (int): internal dimension.
            output_layer (str, optional): Only supports linear or none. Defaults to "linear".
            drop (float, optional): Dropout in attention mechanism. Defaults to 0.0.
            proj_drop (_type_, optional): Dropout after attention output. Defaults to None.
            qkv_bias (bool, optional): Use qkv-bias. Defaults to True.

        """
        super(HydraAttention, self).__init__()
        dropout = proj_drop if proj_drop is not None else drop
        self.dim = dim
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.out = nn.Linear(dim, dim) if output_layer == "linear" else nn.Identity()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, attn_mask=None):
        # x: (B, T, D)
        q, k, v = self.qkv(x).chunk(3, dim=-1)
        q = q / q.norm(dim=-1, keepdim=True)
        k = k / k.norm(dim=-1, keepdim=True)
        if attn_mask is not None:
            k = k.masked_fill(attn_mask.unsqueeze(-1), 0)
        kvw = k * v
        if self.dropout.p > 0:
            kvw = self.dropout(kvw.transpose(-1, -2)).transpose(-1, -2)  # dropout in seq dimension
        out = kvw.sum(dim=-2, keepdim=True) * q
        return self.out(out)


class HydraBlock(Block):
    """Block with Hydra attention."""

    def __init__(self, dim, *args, proj_drop=0.0, drop=None, qkv_bias=False, **kwargs):
        """Create HydraBlock.

        Args:
            dim (int): internal dimension.
            proj_drop (float, optional): dropout after attention output. Defaults to 0.0.
            drop (_type_, optional): dropout in attention calculation. Defaults to None.
            qkv_bias (bool, optional): use qkv-bias. Defaults to False.
            args: additional arguments to pass to the parent class.
            kwargs: additional arguments to pass to the parent class.

        """
        if drop is not None:
            super(HydraBlock, self).__init__(dim, *args, drop=drop, **kwargs)
        else:
            super(HydraBlock, self).__init__(dim, *args, proj_drop=proj_drop, **kwargs)
        self.attn = HydraAttention(dim, proj_drop=proj_drop, drop=drop, qkv_bias=qkv_bias)


class HydraViT(VisionTransformer):
    """Vision Transformer with Hydra attention."""

    def __init__(self,
       img_size=224,
        patch_size=16,
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
        **kwargs,):
        """Create HydraViT model.

        Args:
            args: additional arguments to pass to the parent class.
            kwargs: additional arguments to pass to the parent class.

        """
        super(HydraViT, self).__init__(img_size=img_size,
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
            block_fn=HydraBlock)


@register_model
def hydra_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    """Create Hydra-ViT tiny with patch size 16.

    Args:
        pretrained (bool, optional): UNUSED. Defaults to False.
        img_size (int, optional): Image Resolution. Defaults to 224.
        kwargs: further arguments for the model.

    Returns:
        nn.Module: HydraViT. model.

    """
    size = vit_sizes["Ti"]
    return HydraViT(img_size=img_size, patch_size=16, in_chans=3, **{**size, **kwargs})


@register_model
def hydra_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    """Create Hydra-ViT small with patch size 16.

    Args:
        pretrained (bool, optional): UNUSED. Defaults to False.
        img_size (int, optional): Image Resolution. Defaults to 224.
        kwargs: further arguments for the model.

    Returns:
        nn.Module: HydraViT. model.

    """
    size = vit_sizes["S"]
    return HydraViT(img_size=img_size, patch_size=16, in_chans=3, **{**size, **kwargs})


@register_model
def hydra_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    """Create Hydra-ViT base with patch size 16.

    Args:
        pretrained (bool, optional): UNUSED. Defaults to False.
        img_size (int, optional): Image Resolution. Defaults to 224.
        kwargs: further arguments for the model.

    Returns:
        nn.Module: HydraViT. model.

    """
    size = vit_sizes["B"]
    return HydraViT(img_size=img_size, patch_size=16, in_chans=3, **{**size, **kwargs})


@register_model
def hydra_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    """Create Hydra-ViT large with patch size 16.

    Args:
        pretrained (bool, optional): UNUSED. Defaults to False.
        img_size (int, optional): Image Resolution. Defaults to 224.
        kwargs: further arguments for the model.

    Returns:
        nn.Module: HydraViT. model.

    """
    size = vit_sizes["L"]
    return HydraViT(img_size=img_size, patch_size=16, in_chans=3, **{**size, **kwargs})
