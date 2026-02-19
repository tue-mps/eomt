from functools import partial

import torch
from timm.models import register_model
from torch import nn
from timm.models.layers import Mlp
from timm.models.vision_transformer import Block, PatchEmbed
from timm.models.vision_transformer import (
    VisionTransformer,
)

from models.backbones.utils import vit_sizes

class SwitchMlp(nn.Module):
    """
    ## Routing among multiple FFNs
    taken from https://github.com/labmlai/annotated_deep_learning_paper_implementations/blob/master/labml_nn/transformers/switch/__init__.py
    with some modifications
    """

    def __init__(
        self,
        n_experts,
        dim,
        capacity_factor=1.1,
        drop_tokens=False,
        is_scale_prob=False,
        expert_fn=Mlp,
    ):
        """
        * `capacity_factor` is the capacity of each expert as a factor relative to ideally balanced load
        * `drop_tokens` specifies whether to drop tokens if more tokens are routed to an expert than the capacity
        * `is_scale_prob` specifies whether to multiply the input to the FFN by the routing probability
        * `n_experts` is the number of experts
        * `expert` is the expert layer, a [FFN module](../feed_forward.html)
        * `d_model` is the number of features in a token embedding
        * `d_ff` is the number of features in the hidden layer of the FFN
        * `dropout` is dropout probability in the FFN
        """
        super().__init__()

        self.capacity_factor = capacity_factor
        self.is_scale_prob = is_scale_prob
        self.n_experts = n_experts
        self.drop_tokens = drop_tokens

        # make copies of the FFNs
        self.experts = nn.ModuleList([expert_fn(dim) for _ in range(n_experts)])
        # Routing layer and softmax
        self.switch = nn.Linear(dim, n_experts)
        self.softmax = nn.Softmax(dim=-1)
        self.internal_loss = 0.0
        self.loss_steps = 0

    def update_loss(self, route_counts, route_probs):
        N = route_counts.shape[0]
        T = route_counts.sum().item()
        self.internal_loss += N / T * route_counts @ route_probs
        self.loss_steps += 1

    def get_internal_loss(self, reset=True):
        loss = self.internal_loss / self.loss_steps
        if reset:
            self.internal_loss = 0.0
            self.loss_steps = 0
        return loss

    def forward(self, x):
        """
        * `x` is the input to the switching module with shape `[seq_len, batch_size, d_model]`
        """

        # Capture the shape to change shapes later
        B, N, C = x.shape
        # Flatten the sequence and batch dimensions
        x = x.view(-1, C)

        # Get routing probabilities for each of the tokens.
        # $$p_i(x) = \frac{e^{h(x)_i}}{\sum^N_j e^{h(x)_j}}$$
        # where $N$ is the number of experts `n_experts` and
        # $h(\cdot)$ is the linear transformation of token embeddings.
        with torch.autocast(device_type="cuda", enabled=False):
            # do routing in full precision
            route_prob = self.softmax(self.switch(x.float()))

        # Get the maximum routing probabilities and the routes.
        # We route to the expert with highest probability
        route_prob_max, routes = torch.max(route_prob, dim=-1)

        # Get indexes of tokens going to each expert
        indexes_list = [torch.eq(routes, i).nonzero(as_tuple=True)[0] for i in range(self.n_experts)]

        # Initialize an empty tensor to store outputs
        final_output = torch.zeros_like(x)
        # print(f"x: {x.shape} ({x.dtype}), out: {final_output.shape} ({final_output.dtype})")

        # Capacity of each expert.
        # $$\mathrm{expert\;capacity} =
        # \frac{\mathrm{tokens\;per\;batch}}{\mathrm{number\;of\;experts}}
        # \times \mathrm{capacity\;factor}$$
        capacity = int(self.capacity_factor * len(x) / self.n_experts)
        # Number of tokens routed to each expert.
        counts = x.new_tensor([len(indexes_list[i]) for i in range(self.n_experts)])

        # Initialize an empty list of dropped tokens
        dropped = []
        # Only drop tokens if `drop_tokens` is `True`.
        if self.drop_tokens:
            # Drop tokens in each of the experts
            for i in range(self.n_experts):
                # Ignore if the expert is not over capacity
                if len(indexes_list[i]) <= capacity:
                    continue
                # Shuffle indexes before dropping
                indexes_list[i] = indexes_list[i][torch.randperm(len(indexes_list[i]))]
                # Collect the tokens over capacity as dropped tokens
                dropped.append(indexes_list[i][capacity:])
                # Keep only the tokens upto the capacity of the expert
                indexes_list[i] = indexes_list[i][:capacity]

        # Get outputs of the expert FFNs
        expert_output = [self.experts[i](x[indexes_list[i], :]) for i in range(self.n_experts)]

        final_output = final_output.to(expert_output[0].dtype)
        # Assign to final output
        for i in range(self.n_experts):
            # print(f"expert out: {expert_output[i].shape} ({expert_output[i].dtype}), out: {final_output.shape} ({final_output.dtype})")
            final_output[indexes_list[i], :] = expert_output[i]

        # Pass through the dropped tokens
        if dropped:
            dropped = torch.cat(dropped)
            final_output[dropped, :] = x[dropped, :]

        if self.is_scale_prob:
            # Multiply by the expert outputs by the probabilities $y = p_i(x) E_i(x)$
            final_output = final_output * route_prob_max.view(-1, 1)
        else:
            # Don't scale the values but multiply by $\frac{p}{\hat{p}} = 1$ so that the gradients flow
            # (this is something we experimented with).
            final_output = final_output * (route_prob_max / route_prob_max.detach()).view(-1, 1)

        # Change the shape of the final output back to `[B, N, C]`
        final_output = final_output.view(B, N, C)

        # Return
        #
        # * the final output
        # * number of tokens routed to each expert
        # * sum of probabilities for each expert
        # * number of tokens dropped.
        # * routing probabilities of the selected experts
        #
        # These are used for the load balancing loss and logger
        if self.training:
            self.update_loss(counts, route_prob.mean(0))
        return final_output


class SwitchBlock(Block):
    def __init__(
        self,
        dim,
        num_heads,
        mlp_ratio=4.0,
        act_layer=nn.GELU,
        proj_drop=0.0,
        n_experts=8,
        capacity_factor=1.1,
        **kwargs
    ):
        super().__init__(dim, num_heads, mlp_ratio=mlp_ratio, act_layer=act_layer, proj_drop=proj_drop, **kwargs)
        self.mlp = SwitchMlp(
            n_experts,
            dim,
            capacity_factor=capacity_factor,
            expert_fn=partial(
                Mlp,
                hidden_features=int(dim * mlp_ratio),
                act_layer=act_layer,
                drop=proj_drop,
            ),
        )

    def get_internal_loss(self, reset=True):
        return self.mlp.get_internal_loss(reset)


class SwitchViT(VisionTransformer):
    def __init__(self, img_size=224,
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
        act_layer=None, n_experts=8, capacity_factor=1.1,**kwargs):
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
            block_fn=partial(SwitchBlock, n_experts=n_experts, capacity_factor=capacity_factor)
        )

    def get_internal_loss(self):
        loss = 0.0
        for block in self.blocks:
            loss += block.get_internal_loss(True)
        return loss


@register_model
def switch_8_vit_tiny_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    
    sizes = vit_sizes["Ti"]
    model = SwitchViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        n_experts=8,
        **sizes,
        **kwargs
    )
    return model


@register_model
def switch_8_vit_small_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    
    sizes = vit_sizes["S"]
    model = SwitchViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        n_experts=8,
        **sizes,
        **kwargs
    )
    return model


@register_model
def switch_8_vit_base_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    
    sizes = vit_sizes["B"]
    model = SwitchViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        n_experts=8,
        **sizes,
        **kwargs
    )
    return model


@register_model
def switch_8_vit_large_patch16(pretrained=False, img_size=224, **kwargs):
    if "layer_scale_init_values" in kwargs:
        kwargs["init_values"] = (
            kwargs["layer_scale_init_values"] if "layer_scale" in kwargs and kwargs["layer_scale"] else None
        )
    
    sizes = vit_sizes["L"]
    model = SwitchViT(
        img_size=img_size,
        patch_size=16,
        in_chans=3,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        n_experts=8,
        **sizes,
        **kwargs
    )
    return model
