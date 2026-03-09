# taken from https://github.com/ma-xu/EfficientMod/blob/main/models/EfficientMod.py with slight changes
import torch
from torch import nn
from timm.models.vision_transformer import Block, DropPath
from timm.models import register_model
from utils import to_2tuple


class PatchEmbed(nn.Module):
    def __init__(
        self,
        in_chans=3,
        embed_dim=96,
        patch_size=4,
        patch_stride=4,
        patch_pad=0,
        norm_layer=None,
    ):
        """
        In-shape [b,h,w,c], out-shape[b, h',w',c']
        Args:
            patch_size:
            in_chans:
            embed_dim:
            patch_pad:
            norm_layer:
        """
        super().__init__()
        patch_size = to_2tuple(patch_size)
        self.patch_size = patch_size
        self.in_chans = in_chans
        self.embed_dim = embed_dim
        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_stride,
            padding=patch_pad,
        )
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x.permute(0, 3, 1, 2).contiguous()).permute(0, 2, 3, 1).contiguous()
        if self.norm is not None:
            x = self.norm(x)
        return x


class AttentionBlock(Block):
    def forward(self, x):
        B, H, W, C = x.size()
        x = x.reshape(B, H * W, C).contiguous()
        x = x + self.drop_path1(self.ls1(self.attn(self.norm1(x))))
        x = x + self.drop_path2(self.ls2(self.mlp(self.norm2(x))))
        x = x.reshape(B, H, W, C).contiguous()
        return x


class ContextLayer(nn.Module):
    def __init__(
        self,
        in_dim,
        conv_dim,
        context_size=[3],
        context_act=nn.GELU,
        context_f=True,
        context_g=True,
    ):
        # channel last
        super().__init__()
        self.f = nn.Linear(in_dim, conv_dim) if context_f else nn.Identity()
        self.g = nn.Linear(conv_dim, in_dim) if context_g else nn.Identity()
        self.context_size = context_size
        self.act = context_act() if context_act else nn.Identity()
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size]
        self.context_list = nn.ModuleList()
        for c_size in context_size:
            self.context_list.append(
                nn.Conv2d(
                    conv_dim,
                    conv_dim,
                    c_size,
                    stride=1,
                    padding=c_size // 2,
                    groups=conv_dim,
                )
            )

    def forward(self, x):
        x = self.f(x).permute(0, 3, 1, 2).contiguous()
        out = 0
        for i in range(len(self.context_list)):
            ctx = self.act(self.context_list[i](x))
            out = out + ctx
        out = self.g(out.permute(0, 2, 3, 1).contiguous())
        return out


class ExpandMlp(nn.Module):
    def __init__(
        self,
        in_features,
        hidden_features=None,
        out_features=None,
        act_layer=nn.GELU,
        drop=0.0,
        bias=True,
        conv_in_mlp=True,
        conv_group_dim=4,
        context_size=3,
        context_act=nn.GELU,
        context_f=True,
        context_g=True,
    ):
        # channel last
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features, bias=bias)
        self.conv_in_mlp = conv_in_mlp
        if self.conv_in_mlp:
            self.conv_group_dim = conv_group_dim
            self.conv_dim = hidden_features // conv_group_dim
            self.context_layer = ContextLayer(
                in_features,
                self.conv_dim,
                context_size,
                context_act,
                context_f,
                context_g,
            )

        self.fc2 = nn.Linear(hidden_features, out_features, bias=bias)

        if hidden_features == in_features and conv_group_dim == 1:
            self.expand_dim = False
        else:
            self.expand_dim = True
            self.act = act_layer()
            self.drop = nn.Dropout(drop)

    def forward(self, x):
        if self.conv_in_mlp:
            conv_x = self.context_layer(x)
        x = self.fc1(x)
        if self.expand_dim:
            x = self.act(x)
            x = self.drop(x)
        if self.conv_in_mlp:
            if self.expand_dim:
                x = x * conv_x.repeat(1, 1, 1, self.conv_group_dim)
            else:
                x = x * conv_x
        x = self.fc2(x)
        return x


class BasicBlock(nn.Module):
    def __init__(
        self,
        dim,
        mlp_ratio=4.0,
        conv_in_mlp=True,
        drop=0.0,
        drop_path=0.0,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        bias=True,
        use_layerscale=False,
        layerscale_value=1e-4,
        conv_group_dim=4,
        context_size=3,
        context_act=nn.GELU,
        context_f=True,
        context_g=True,
    ):
        super().__init__()
        self.norm = norm_layer(dim)
        self.mlp = ExpandMlp(
            in_features=dim,
            hidden_features=int(dim * mlp_ratio),
            act_layer=act_layer,
            drop=drop,
            bias=bias,
            conv_in_mlp=conv_in_mlp,
            conv_group_dim=conv_group_dim,
            context_size=context_size,
            context_act=context_act,
            context_f=context_f,
            context_g=context_g,
        )
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.gamma_1 = 1.0
        if use_layerscale:
            self.gamma_1 = nn.Parameter(layerscale_value * torch.ones((dim)), requires_grad=True)

    def forward(self, x):
        shortcut = x
        x = shortcut + self.drop_path(self.gamma_1 * self.mlp(self.norm(x)))
        return x


class BasicLayer(nn.Module):
    def __init__(
        self,
        dim,
        out_dim,
        depth,
        mlp_ratio=4.0,
        att_ratio=4.0,
        conv_in_mlp=True,
        drop=0.0,
        drop_path=0.0,
        num_heads=8,
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        bias=True,
        use_layerscale=False,
        layerscale_value=1e-4,
        conv_group_dim=4,
        context_size=3,
        context_act=nn.GELU,
        context_f=True,
        context_g=True,
        downsample=None,
        patch_size=3,
        patch_stride=2,
        patch_pad=1,
        patch_norm=True,
        attention_depth=0,
    ):

        super().__init__()
        self.dim = dim
        self.depth = depth
        if not isinstance(mlp_ratio, (list, tuple)):
            mlp_ratio = [mlp_ratio] * depth
        if not isinstance(conv_group_dim, (list, tuple)):
            conv_group_dim = [conv_group_dim] * depth
        if not isinstance(context_size, (list, tuple)):
            context_size = [context_size] * depth
        # build blocks
        self.blocks = nn.ModuleList(
            [
                BasicBlock(
                    dim=dim,
                    mlp_ratio=mlp_ratio[i],
                    conv_in_mlp=conv_in_mlp,
                    drop=drop,
                    drop_path=(drop_path[i] if isinstance(drop_path, list) else drop_path),
                    act_layer=act_layer,
                    norm_layer=norm_layer,
                    bias=bias,
                    use_layerscale=use_layerscale,
                    layerscale_value=layerscale_value,
                    conv_group_dim=conv_group_dim[i],
                    context_size=context_size[i],
                    context_act=context_act,
                    context_f=context_f,
                    context_g=context_g,
                )
                for i in range(depth)
            ]
        )

        if attention_depth > 0:
            for j in range(attention_depth):
                try:
                    self.blocks.append(
                        AttentionBlock(
                            dim=dim,
                            mlp_ratio=att_ratio,
                            attn_drop=drop,
                            proj_drop=drop,
                            drop_path=drop_path[depth + j],
                            num_heads=num_heads,
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                        )
                    )
                except TypeError:
                    self.blocks.append(
                        AttentionBlock(
                            dim=dim,
                            mlp_ratio=att_ratio,
                            attn_drop=drop,
                            drop=drop,
                            drop_path=drop_path[depth + j],
                            num_heads=num_heads,
                            act_layer=act_layer,
                            norm_layer=norm_layer,
                        )
                    )

        if downsample is not None:
            self.downsample = downsample(
                in_chans=dim,
                embed_dim=out_dim,
                patch_size=patch_size,
                patch_stride=patch_stride,
                patch_pad=patch_pad,
                norm_layer=norm_layer if patch_norm else None,
            )
        else:
            self.downsample = None

    def forward(self, x):
        for blk in self.blocks:
            x = blk(x)
        if self.downsample is not None:
            x = self.downsample(x)
        return x


class EfficientMod(nn.Module):
    def __init__(
        self,
        in_chans=3,
        num_classes=1000,
        # downsize patch-related
        patch_size=[4, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_pad=[0, 1, 1, 1],
        patch_norm=True,
        # newwork configuration
        embed_dim=[64, 128, 256, 512],
        depths=[2, 2, 6, 2],
        attention_depth=[0, 0, 0, 0],
        mlp_ratio=[4.0, 4.0, 4.0, 4.0],
        att_ratio=[4, 4, 4, 4],
        num_heads=8,
        conv_in_mlp=[True, True, True, True],
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_layerscale=False,
        layerscale_value=1e-4,
        bias=True,
        drop_rate=0.0,
        drop_path_rate=0.0,
        conv_group_dim=[4, 4, 4, 4],
        context_size=[3, 3, 3, 3],
        context_act=nn.GELU,
        context_f=True,
        context_g=True,
        **kwargs
    ):
        super().__init__()

        self.num_layers = len(depths)

        self.depths = depths
        self.attention_depth = attention_depth
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        self.patch_norm = patch_norm
        self.num_features = embed_dim[-1]

        # split image into patches using either non-overlapped embedding or overlapped embedding
        self.patch_embed = PatchEmbed(
            in_chans=in_chans,
            embed_dim=embed_dim[0],
            patch_size=patch_size[0],
            patch_stride=patch_stride[0],
            patch_pad=patch_pad[0],
            norm_layer=norm_layer if self.patch_norm else None,
        )

        # stochastic depth
        dpr = [
            x.item() for x in torch.linspace(0, drop_path_rate, (sum(depths) + sum(attention_depth)))
        ]  # stochastic depth decay rule

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(
                dim=embed_dim[i_layer],
                out_dim=(embed_dim[i_layer + 1] if (i_layer < self.num_layers - 1) else None),
                depth=depths[i_layer],
                mlp_ratio=mlp_ratio[i_layer],
                att_ratio=att_ratio[i_layer],
                conv_in_mlp=conv_in_mlp[i_layer],
                drop=drop_rate,
                num_heads=num_heads,
                drop_path=dpr[
                    sum(depths[:i_layer])
                    + sum(attention_depth[:i_layer]) : sum(depths[: i_layer + 1])
                    + sum(attention_depth[: i_layer + 1])
                ],
                act_layer=act_layer,
                norm_layer=norm_layer,
                bias=bias,
                use_layerscale=use_layerscale,
                layerscale_value=layerscale_value,
                conv_group_dim=conv_group_dim[i_layer],
                context_size=context_size[i_layer],
                context_act=context_act,
                context_f=context_f,
                context_g=context_g,
                downsample=PatchEmbed if (i_layer < self.num_layers - 1) else None,
                patch_size=(patch_size[i_layer + 1] if (i_layer < self.num_layers - 1) else None),
                patch_stride=(patch_stride[i_layer + 1] if (i_layer < self.num_layers - 1) else None),
                patch_pad=(patch_pad[i_layer + 1] if (i_layer < self.num_layers - 1) else None),
                patch_norm=patch_norm,
                attention_depth=attention_depth[i_layer],
            )
            self.layers.append(layer)

        self.norm = norm_layer(self.num_features)
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.head = nn.Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        self.embed_dim = self.num_features

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear or nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        x = self.patch_embed(x.permute(0, 2, 3, 1))
        for layer in self.layers:
            x = layer(x)
        x = self.norm(x)
        x = self.avgpool(x.permute(0, 3, 1, 2))
        x = torch.flatten(x, 1)
        return x

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

    def set_image_res(self, *args, **kwargs):
        pass


@register_model
def efficient_mod_xxs(pretrained=False, num_classes=1000, **kwargs):
    depths = [2, 2, 6, 2]
    attention_depth = [0, 0, 1, 2]
    att_ratio = [0, 0, 4, 4]
    mlp_ratio = [
        [1, 6, 1, 6],
        [1, 6, 1, 6],
        [1, 6] * 3,
        [1, 6, 1, 6],
    ]
    context_size = [
        [7] * 10,
        [7] * 10,
        [7] * 20,
        [7] * 10,
    ]
    conv_group_dim = mlp_ratio

    heads = kwargs.get("num_heads", None)
    if heads is None:
        heads = 8

    drop_rate = kwargs["proj_drop"] if "proj_drop" in kwargs else (kwargs["dropout"] if "dropout" in kwargs else 0.0)

    model = EfficientMod(
        in_chans=3,
        num_classes=num_classes,
        num_heads=heads,
        patch_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_pad=[3, 1, 1, 1],
        patch_norm=True,
        embed_dim=[32, 64, 128, 256],
        depths=depths,
        attention_depth=attention_depth,
        mlp_ratio=mlp_ratio,
        att_ratio=att_ratio,
        conv_in_mlp=[True, True, True, True],
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_layerscale=kwargs["layer_scale"],
        layerscale_value=kwargs["layer_scale_init_values"],
        bias=kwargs["qkv_bias"],
        drop_rate=drop_rate,
        drop_path_rate=kwargs["drop_path_rate"],
        conv_group_dim=conv_group_dim,
        context_size=context_size,
        context_act=nn.GELU,
        context_f=True,
        context_g=True,
    )
    return model


@register_model
def efficient_mod_xs(pretrained=False, num_classes=1000, **kwargs):
    depths = [3, 3, 4, 2]
    attention_depth = [0, 0, 3, 3]
    att_ratio = [4, 4, 4, 4]
    mlp_ratio = [
        [1, 4, 1, 4] * 4,
        [1, 4, 1, 4] * 4,
        [1, 4, 1, 4] * 10,
        [1, 4, 1, 4] * 4,
    ]
    context_size = [
        [7] * 10,
        [7] * 10,
        [7] * 20,
        [7] * 10,
    ]
    conv_group_dim = mlp_ratio

    heads = kwargs.get("num_heads", None)
    if heads is None:
        heads = 8

    drop_rate = kwargs["proj_drop"] if "proj_drop" in kwargs else (kwargs["dropout"] if "dropout" in kwargs else 0.0)

    model = EfficientMod(
        in_chans=3,
        num_classes=num_classes,
        num_heads=heads,
        patch_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_pad=[3, 1, 1, 1],
        patch_norm=True,
        embed_dim=[32, 64, 128, 256],
        depths=depths,
        attention_depth=attention_depth,
        mlp_ratio=mlp_ratio,
        att_ratio=att_ratio,
        conv_in_mlp=[True, True, True, True],
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_layerscale=kwargs["layer_scale"],
        layerscale_value=kwargs["layer_scale_init_values"],
        bias=kwargs["qkv_bias"],
        drop_rate=drop_rate,
        drop_path_rate=kwargs["drop_path_rate"],
        conv_group_dim=conv_group_dim,
        context_size=context_size,
        context_act=nn.GELU,
        context_f=True,
        context_g=True,
    )
    return model


@register_model
def efficient_mod_s(pretrained=False, num_classes=1000, **kwargs):
    depths = [4, 4, 8, 4]
    attention_depth = [0, 0, 4, 4]
    att_ratio = [4, 4, 4, 5]
    mlp_ratio = [
        [1, 6, 1, 6] * 4,
        [1, 6, 1, 6] * 4,
        [1, 6, 1, 6] * 10,
        [1, 6, 1, 6] * 4,
    ]
    context_size = [
        [7] * 10,
        [7] * 10,
        [7] * 20,
        [7] * 10,
    ]
    conv_group_dim = mlp_ratio

    heads = kwargs.get("num_heads", None)
    if heads is None:
        heads = 8

    drop_rate = kwargs["proj_drop"] if "proj_drop" in kwargs else (kwargs["dropout"] if "dropout" in kwargs else 0.0)

    model = EfficientMod(
        in_chans=3,
        num_classes=num_classes,
        num_heads=heads,
        patch_size=[7, 3, 3, 3],
        patch_stride=[4, 2, 2, 2],
        patch_pad=[3, 1, 1, 1],
        patch_norm=True,
        embed_dim=[32, 64, 128, 256],
        depths=depths,
        attention_depth=attention_depth,
        mlp_ratio=mlp_ratio,
        att_ratio=att_ratio,
        conv_in_mlp=[True, True, True, True],
        act_layer=nn.GELU,
        norm_layer=nn.LayerNorm,
        use_layerscale=kwargs["layer_scale"],
        layerscale_value=kwargs["layer_scale_init_values"],
        bias=kwargs["qkv_bias"],
        drop_rate=drop_rate,
        drop_path_rate=kwargs["drop_path_rate"],
        conv_group_dim=conv_group_dim,
        context_size=context_size,
        context_act=nn.GELU,
        context_f=True,
        context_g=True,
    )
    return model
