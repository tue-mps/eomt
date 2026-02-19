

from typing import Optional
import timm
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch
import os
from models.backbones.convViT import to_doubletuple
import torch.nn as nn


class ConvViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="cvt_13",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            no_embed_class = True,
            num_classes=0
        )
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print("checkpoint found at {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path,weights_only=False)
            
            state_dict = None
            if 'model_state' in checkpoint:
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state'].items()}
                
            missing, unexpected = self.backbone.load_state_dict(state_dict, strict=False) if state_dict else self.backbone.load_state_dict(checkpoint, strict=False)
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
                
         
        self.backbone.set_image_res(img_size)
        self.num_heads = [1, 3, 6]
        self.embed_dim = 384
        self.patch_size = to_doubletuple(3)
        self.grid_size = get_cvt13_final_grid(img_size[0])
        self.num_prefix_tokens = 0
        self.depths = [1, 2, 10]
        stages = [getattr(self.backbone, f'stage{i}') for i in range(self.backbone.num_stages)]
        depth_prefix_sum_list = [0]
        for d in self.depths:
            depth_prefix_sum_list.append(depth_prefix_sum_list[-1]+d)
        depth_prefix_sum_list_pre = depth_prefix_sum_list[:-1]
        self.depth_prefix_sum_list_post = depth_prefix_sum_list[1:]
        self.patch_embed_dict = nn.ModuleDict({str(d):layer.patch_embed for d,layer in zip(depth_prefix_sum_list_pre,stages)})
        self.blocks = [block for layer in stages for block in layer.blocks]
        self.norm = self.backbone.norm
        self.attn_multiplier = multiplier
        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        
        return x
    
    def q_mlp(self, block, q: torch.Tensor):
        q = q + block.drop_path(block.mlp(block.norm2(q)))
        return q
    
    def run_block(self, block, x: torch.Tensor, curr : int):
        idx = str(curr)
        
        if idx in self.patch_embed_dict.keys() and self.patch_embed_dict[idx] is not None:
            x = self.patch_embed_dict[idx](x)
            self.B, self.C, self.H, self.W = x.size()
            x = x.permute(0, 2, 3, 1).reshape(self.B, self.H*self.W, self.C)
        x = block(x,self.H,self.W)
        if (curr+1) in self.depth_prefix_sum_list_post:
            x = x.reshape(self.B, self.H, self.W, self.C).permute(0, 3, 1, 2)
        return x
    
def get_cvt13_final_grid(input_resolution):
    """
    Calculate final grid size for CvT-13 from input resolution.
    
    Args:
        input_resolution (int): Input image resolution (assumes square images)
    
    Returns:
        tuple: (height, width) of final grid
    """
    # Stage 1: kernel=7, stride=4, padding=2
    h = (input_resolution + 4 - 7) // 4 + 1
    
    # Stage 2: kernel=3, stride=2, padding=1
    h = (h + 2 - 3) // 2 + 1
    
    # Stage 3: kernel=3, stride=2, padding=1
    h = (h + 2 - 3) // 2 + 1
    
    return (h, h)
