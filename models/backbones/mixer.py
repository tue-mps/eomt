from functools import partial

from timm.models import register_model
from timm.models.mlp_mixer import MixerBlock, Mlp, MlpMixer, PatchEmbed
from torch import nn

class Mixer(MlpMixer):
    """MLP-Mixer model.

    Uses only MLPs and no self-attention.
    """

    def __init__(
        self,
        num_classes=1000,
        img_size=224,
        in_chans=3,
        patch_size=16,
        num_blocks=8,
        embed_dim=512,
        mlp_ratio=(0.5, 4.0),
        block_layer=MixerBlock,
        mlp_layer=Mlp,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        act_layer=nn.GELU,
        drop_rate=0.0,
        drop_path_rate=0.0,
        nlhb=False,
        stem_norm=False,
        global_pool="avg",
        **kwargs
    ):
        """Create MLP-Mixer model.

        Args:
            num_classes (int, optional): Number of classes. Defaults to 1000.
            img_size (int, optional): Image resolution. Defaults to 224.
            in_chans (int, optional): Number of input channels in image. Defaults to 3.
            patch_size (int, optional): Patch size. Defaults to 16.
            num_blocks (int, optional): Model depth. Defaults to 8.
            embed_dim (int, optional): Embedding dimension. Defaults to 512.
            mlp_ratio (tuple, optional): Ratio of MLP intermediate dimensions. Defaults to (0.5, 4.0).
            block_layer (nn.Module, optional): Block layer. Defaults to MixerBlock.
            mlp_layer (nn.Module, optional): MLP layer. Defaults to Mlp.
            norm_layer (nn.Module, optional): Normalization layer. Defaults to partial(nn.LayerNorm, eps=1e-6).
            act_layer (nn.Module, optional): Activation function. Defaults to nn.GELU.
            drop_rate (float, optional): Dropout rate. Defaults to 0.0.
            drop_path_rate (float, optional): Drop path rate. Defaults to 0.0.
            nlhb (bool, optional): Use NLHB initialization. Defaults to False.
            stem_norm (bool, optional): Use normalization in stem. Defaults to False.
            global_pool (str, optional): Global pooling type. Defaults to "avg".
            kwargs: Additional arguments for the model.

        """
        super().__init__(
            num_classes,
            img_size,
            in_chans,
            patch_size,
            num_blocks,
            embed_dim,
            mlp_ratio,
            block_layer,
            mlp_layer,
            norm_layer,
            act_layer,
            drop_rate,
            drop_path_rate,
            nlhb,
            stem_norm,
            global_pool,
        )
        self.img_size = img_size
        self.embed_dim = embed_dim
        self.nlhb = nlhb

        self.stem = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
            norm_layer=norm_layer if stem_norm else None,
        )

    def set_image_res(self, res):
        if res == self.img_size:
            return
        raise NotImplementedError("Changing resolution for mixer is not implemented.")


@register_model
def mixer_s32(pretrained=False, img_size=224, **kwargs):
    """Mixer-S/32 224x224.

    Paper: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if "pretrained_cfg" in kwargs:
        kwargs.pop("pretrained_cfg")
    model_args = dict(patch_size=32, num_blocks=8, embed_dim=512, **kwargs)
    return Mixer(img_size=img_size, **model_args)


@register_model
def mixer_s16(pretrained=False, img_size=224, **kwargs):
    """Mixer-S/16 224x224.

    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601.
    """
    if "pretrained_cfg" in kwargs:
        kwargs.pop("pretrained_cfg")
    model_args = dict(patch_size=16, num_blocks=8, embed_dim=512, **kwargs)
    return Mixer(img_size=img_size, **model_args)


@register_model
def mixer_b32(pretrained=False, img_size=224, **kwargs):
    """Mixer-B/32 224x224.

    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if "pretrained_cfg" in kwargs:
        kwargs.pop("pretrained_cfg")
    model_args = dict(patch_size=32, num_blocks=12, embed_dim=768, **kwargs)
    return Mixer(img_size=img_size, **model_args)


@register_model
def mixer_b16(pretrained=False, img_size=224, **kwargs):
    """Mixer-B/16 224x224.

    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if "pretrained_cfg" in kwargs:
        kwargs.pop("pretrained_cfg")
    model_args = dict(patch_size=16, num_blocks=12, embed_dim=768, **kwargs)
    return Mixer(img_size=img_size, **model_args)


@register_model
def mixer_l32(pretrained=False, img_size=224, **kwargs):
    """Mixer-L/32 224x224.

    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if "pretrained_cfg" in kwargs:
        kwargs.pop("pretrained_cfg")
    model_args = dict(patch_size=32, num_blocks=24, embed_dim=1024, **kwargs)
    return Mixer(img_size=img_size, **model_args)


@register_model
def mixer_l16(pretrained=False, img_size=224, **kwargs):
    """Mixer-L/16 224x224.

    Paper:  'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    if "pretrained_cfg" in kwargs:
        kwargs.pop("pretrained_cfg")
    model_args = dict(patch_size=16, num_blocks=24, embed_dim=1024, **kwargs)
    return Mixer(img_size=img_size, **model_args)
