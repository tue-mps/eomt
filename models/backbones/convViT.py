# taken from https://github.com/microsoft/CvT/blob/main/lib/models/cls_cvt.py with slight modifications

from collections import OrderedDict
import torch

from timm.models import register_model
from timm.models.layers import DropPath, Mlp, to_2tuple
from torch import nn
from torch.nn import functional as F
from torch.nn.init import trunc_normal_

to_doubletuple = to_2tuple

class RearrangeBCHWtoSequence(nn.Module):
    def forward(self, x):
        return x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2]*x.shape[3], x.shape[1])



class CvTAttention(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        qkv_bias=False,
        attn_drop=0.0,
        proj_drop=0.0,
        method="dw_bn",
        kernel_size=3,
        stride_kv=1,
        stride_q=1,
        padding_kv=1,
        padding_q=1,
        with_cls_token=True,
        **kwargs,
    ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.dim = dim_out
        self.num_heads = num_heads
        # head_dim = self.qkv_dim // num_heads
        self.scale = dim_out**-0.5
        self.with_cls_token = with_cls_token

        self.conv_proj_q = self._build_projection(
            dim_in,
            dim_out,
            kernel_size,
            padding_q,
            stride_q,
            "linear" if method == "avg" else method,
        )
        self.conv_proj_k = self._build_projection(dim_in, dim_out, kernel_size, padding_kv, stride_kv, method)
        self.conv_proj_v = self._build_projection(dim_in, dim_out, kernel_size, padding_kv, stride_kv, method)

        self.proj_q = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_k = nn.Linear(dim_in, dim_out, bias=qkv_bias)
        self.proj_v = nn.Linear(dim_in, dim_out, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim_out, dim_out)
        self.proj_drop = nn.Dropout(proj_drop)

    def _build_projection(self, dim_in, dim_out, kernel_size, padding, stride, method):
        if method == "dw_bn":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "conv",
                            nn.Conv2d(
                                dim_in,
                                dim_in,
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                bias=False,
                                groups=dim_in,
                            ),
                        ),
                        ("bn", nn.BatchNorm2d(dim_in)),
                        ("rearrange", RearrangeBCHWtoSequence()),
                    ]
                )
            )
        elif method == "avg":
            proj = nn.Sequential(
                OrderedDict(
                    [
                        (
                            "avg",
                            nn.AvgPool2d(
                                kernel_size=kernel_size,
                                padding=padding,
                                stride=stride,
                                ceil_mode=True,
                            ),
                        ),
                        ("rearrange", RearrangeBCHWtoSequence()),
                    ]
                )
            )
        elif method == "linear":
            proj = None
        else:
            raise ValueError("Unknown method ({})".format(method))

        return proj

    def forward_conv(self, x, h, w):
        if self.with_cls_token:
            cls_token, x = torch.split(x, [1, h * w], 1)

        x = x.reshape(x.shape[0], h, w, x.shape[2]).permute(0, 3, 1, 2)

        if self.conv_proj_q is not None:
            q = self.conv_proj_q(x)
        else:
            q = x.permute(0, 2, 3, 1).reshape(x.shape[0], h*w, x.shape[1])

        if self.conv_proj_k is not None:
            k = self.conv_proj_k(x)
        else:
            k = x.permute(0, 2, 3, 1).reshape(x.shape[0], h*w, x.shape[1])

        if self.conv_proj_v is not None:
            v = self.conv_proj_v(x)
        else:
            v = x.permute(0, 2, 3, 1).reshape(x.shape[0], h*w, x.shape[1])

        if self.with_cls_token:
            q = torch.cat((cls_token, q), dim=1)
            k = torch.cat((cls_token, k), dim=1)
            v = torch.cat((cls_token, v), dim=1)

        return q, k, v
    
    def forward(self, x, h, w):
        if self.conv_proj_q is not None or self.conv_proj_k is not None or self.conv_proj_v is not None:
            q, k, v = self.forward_conv(x, h, w)

        
        q = self.proj_q(q)
        q = q.reshape(q.shape[0], q.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        
        k = self.proj_k(k)
        k = k.reshape(k.shape[0], k.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)
        
        v = self.proj_v(v)
        v = v.reshape(v.shape[0], v.shape[1], self.num_heads, -1).permute(0, 2, 1, 3)

        attn_score = torch.einsum("bhlk,bhtk->bhlt", [q, k]) * self.scale
        attn = F.softmax(attn_score, dim=-1)
        attn = self.attn_drop(attn)

        x = torch.einsum("bhlt,bhtv->bhlv", [attn, v])
        
        x = x.permute(0, 2, 1, 3).reshape(x.shape[0], x.shape[2], -1)

        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class CvTBlock(nn.Module):
    def __init__(
        self,
        dim_in,
        dim_out,
        num_heads,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop=0.0,
        attn_drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        **kwargs,
    ):
        super().__init__()

        self.with_cls_token = kwargs["with_cls_token"]

        self.norm1 = norm_layer(dim_in)
        self.attn = CvTAttention(dim_in, dim_out, num_heads, qkv_bias, attn_drop, drop, **kwargs)

        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim_out)

        dim_mlp_hidden = int(dim_out * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim_out,
            hidden_features=dim_mlp_hidden,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, h, w):
        res = x

        x = self.norm1(x)
        attn = self.attn(x, h, w)
        x = res + self.drop_path(attn)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


class ConvEmbed(nn.Module):
    """Image to Conv Embedding"""

    def __init__(
        self,
        img_size=224,
        patch_size=7,
        in_chans=3,
        embed_dim=64,
        stride=4,
        padding=2,
        norm_layer=None,
    ):
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        img_size = to_2tuple(img_size)
        self.grid_size = tuple([s // p for s, p in zip(img_size, self.patch_size)])
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=padding)
        self.norm = norm_layer(embed_dim) if norm_layer else None

    def forward(self, x):
        x = self.proj(x)

        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)
        if self.norm:
            x = self.norm(x)
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)

        return x


class VisionTransformer(nn.Module):
    """Vision Transformer with support for patch or hybrid CNN input stage"""

    def __init__(
        self,
        img_size=224,
        patch_size=16,
        patch_stride=16,
        patch_padding=0,
        in_chans=3,
        embed_dim=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        qkv_bias=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        **kwargs,
    ):
        super().__init__()
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models

        self.rearrage = None

        self.patch_embed = ConvEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            stride=patch_stride,
            padding=patch_padding,
            embed_dim=embed_dim,
            norm_layer=norm_layer,
        )

        with_cls_token = kwargs["with_cls_token"]
        if with_cls_token:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        else:
            self.cls_token = None

        self.pos_drop = nn.Dropout(p=drop_rate)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        blocks = []
        for j in range(depth):
            blocks.append(
                CvTBlock(
                    dim_in=embed_dim,
                    dim_out=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[j],
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    **kwargs,
                )
            )
        self.blocks = nn.ModuleList(blocks)

        if self.cls_token is not None:
            trunc_normal_(self.cls_token, std=0.02)

        if init == "xavier":
            self.apply(self._init_weights_xavier)
        else:
            self.apply(self._init_weights_trunc_normal)

    def _init_weights_trunc_normal(self, m):
        if isinstance(m, nn.Linear):
            # logger.info('=> init weight of Linear from trunc norm')
            trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                # logger.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _init_weights_xavier(self, m):
        if isinstance(m, nn.Linear):
            # logger.info('=> init weight of Linear from xavier uniform')
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                # logger.info('=> init bias of Linear to zeros')
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        B, C, H, W = x.size()

        x = x.permute(0, 2, 3, 1).reshape(B, H*W, C)

        cls_tokens = None
        if self.cls_token is not None:
            # stole cls_tokens impl from Phil Wang, thanks
            cls_tokens = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_tokens, x), dim=1)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x, H, W)

        if self.cls_token is not None:
            cls_tokens, x = torch.split(x, [1, H * W], 1)            
        x = x.reshape(B, H, W, C).permute(0, 3, 1, 2)


        return x, cls_tokens


class ConvolutionalVisionTransformer(nn.Module):
    def __init__(
        self,
        img_size,
        patch_size=[7, 3, 3],
        in_chans=3,
        num_heads=[1, 3, 6],
        num_classes=1000,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        init="trunc_norm",
        patch_stride=[4, 2, 2],
        patch_padding=[2, 1, 1],
        embed_dim=[64, 192, 384],
        depth=[1, 4, 16],
        mlp_ratio=4.0,
        qkv_bias=True,
        cls_token=[False, False, False],
        drop_rate=0.0,
        drop_path_rate=0.0,
        attn_drop_rate=0.0,
        method="dw_bn",
        kernel_size=3,
        padding_q=1,
        padding_kv=1,
        stride_q=1,
        stride_kv=2,
        num_stages=3,
        **kwargs,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.num_stages = num_stages
        stage_kwargs = {
            "patch_size": patch_size,
            "patch_stride": patch_stride,
            "patch_padding": patch_padding,
            "embed_dim": embed_dim,
            "depth": depth,
            "num_heads": num_heads,
            "mlp_ratio": mlp_ratio,
            "qkv_bias": qkv_bias,
            "drop_rate": drop_rate,
            "attn_drop_rate": attn_drop_rate,
            "drop_path_rate": drop_path_rate,
            "with_cls_token": cls_token,
            "method": method,
            "kernel_size": kernel_size,
            "padding_q": padding_q,
            "padding_kv": padding_kv,
            "stride_kv": stride_kv,
            "stride_q": stride_q,
        }
        for val in stage_kwargs.values():
            assert not isinstance(val, list) or len(val) == num_stages, (
                "Argument must be value or list of values for each stage. "
                f"Found {len(val)} values for {num_stages} stages."
            )

        for i in range(self.num_stages):
            kwargs = {key: val[i] if isinstance(val, list) else val for key, val in stage_kwargs.items()}

            stage = VisionTransformer(
                img_size=img_size,
                in_chans=in_chans,
                init=init,
                act_layer=act_layer,
                norm_layer=norm_layer,
                **kwargs,
            )
            setattr(self, f"stage{i}", stage)

            in_chans = embed_dim[i] if isinstance(embed_dim, list) else embed_dim

        self.embed_dim = embed_dim[-1] if isinstance(embed_dim, list) else embed_dim
        self.norm = norm_layer(self.embed_dim)
        self.cls_token = cls_token

        # Classifier head
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        if not isinstance(self.head, nn.Identity):
            trunc_normal_(self.head.weight, std=0.02)

    def set_image_res(self, res):
        return

    @torch.jit.ignore
    def no_weight_decay(self):
        layers = set()
        for i in range(self.num_stages):
            layers.add(f"stage{i}.pos_embed")
            layers.add(f"stage{i}.cls_token")

        return layers

    def forward_features(self, x):
        for i in range(self.num_stages):
            x, cls_tokens = getattr(self, f"stage{i}")(x)

        if self.cls_token:
            x = self.norm(cls_tokens)
            x = torch.squeeze(x)
        else:
            # rearrange(x, "b c h w -> b (h w) c")
            x = x.permute(0, 2, 3, 1).reshape(x.shape[0], x.shape[2]*x.shape[3], x.shape[1])
            x = self.norm(x)
            x = torch.mean(x, dim=1)

        return x


    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)

        return x


@register_model
def cvt_13(img_size=224, **kwargs):
    model = ConvolutionalVisionTransformer(img_size,patch_size=[7, 3, 3], depth=[1, 2, 10], pos_embed=False, **kwargs)
    return model


@register_model
def cvt_21(img_size=224, **kwargs):
    model = ConvolutionalVisionTransformer(patch_size=[7, 3, 3], depth=[1, 4, 16], pos_embed=False, **kwargs)
    return model


@register_model
def cvt_w24(img_size=224, **kwargs):
    model = ConvolutionalVisionTransformer(
        patch_size=[7, 3, 3],
        depth=[2, 2, 20],
        embed_dim=[192, 768, 1024],
        num_heads=[3, 12, 16],
        pos_embed=False,
        **kwargs,
    )
    return model
