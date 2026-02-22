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
        # Backbone creation
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            no_embed_class=True,
            num_classes=0
        )
        
        # Checkpoint loading
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print("checkpoint found at {}".format(ckpt_path))
            checkpoint = torch.load(ckpt_path, weights_only=False)
            
            state_dict = None
            if 'model_state' in checkpoint:
                state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state'].items()}
                
            missing, unexpected = self.backbone.load_state_dict(
                state_dict if state_dict else checkpoint, strict=False
            )
            print(f"Missing keys: {missing}")
            print(f"Unexpected keys: {unexpected}")
                
        # Configuration
        self.backbone.set_image_res(img_size)
        self.num_heads = [1, 3, 6]
        self.embed_dim = 384
        self.patch_size = to_doubletuple(3)
        self.grid_size = get_cvt13_final_grid(img_size[0])
        self.num_prefix_tokens = 0
        self.depths = [1, 2, 10]
        
        # Stage and Block Setup
        stages = [getattr(self.backbone, f'stage{i}') for i in range(self.backbone.num_stages)]
        
        depth_prefix_sum_list = [0]
        for d in self.depths:
            depth_prefix_sum_list.append(depth_prefix_sum_list[-1]+d)
            
        depth_prefix_sum_list_pre = depth_prefix_sum_list[:-1]
        self.depth_prefix_sum_list_post = depth_prefix_sum_list[1:]
        
        # Mappings
        self.patch_embed_dict = nn.ModuleDict({
            str(d): layer.patch_embed for d, layer in zip(depth_prefix_sum_list_pre, stages)
        })
        self.blocks = [block for layer in stages for block in layer.blocks]
        self.norm = self.backbone.norm
        self.attn_multiplier = multiplier
        
        # Normalization constants
        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)

    def pre_block(self, x: torch.Tensor):
        return x
    
    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i):
        idx = str(i)
        
        # 1. Patch Embedding / Downsampling (ConvViT specific)
        if idx in self.patch_embed_dict.keys() and self.patch_embed_dict[idx] is not None:
            x = self.patch_embed_dict[idx](x)
            self.B, self.C, self.H, self.W = x.size()
            x = x.permute(0, 2, 3, 1).reshape(self.B, self.H*self.W, self.C)
            
        # 2. Block Execution
        if i >= len(self.blocks) - eomt_obj.num_blocks:
            # --- EoMT Interaction Logic ---
            
            # Prepare combined tokens for norm/prediction
            xq = torch.cat((q[None, :, :].expand(x.shape[0], -1, -1), x), dim=1)
            
            # Pre-Attn Norm (using block.norm1)
            pre_attn = block.norm1(xq)
            
            # Split for prediction
            x_norm, q_norm = pre_attn[:, eomt_obj.num_q:, :], pre_attn[:, : eomt_obj.num_q, :]
            
            # EoMT Predictions
            mask_logits, class_logits = eomt_obj._predict(x_norm, q_norm)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)

            # EoMT Cross-Attention
            after_eomt = torch.cat((q_norm, x_norm), dim=1)
            new_x = eomt_obj.attn[i - len(self.blocks)](self.norm(after_eomt))
            
            # Update q/x with Cross-Attn results
            # Note: ConvViT backbone does NOT have ls_list (LayerScale), using identity implicitly
            ls_val = eomt_obj.ls_list[i - len(self.blocks)](new_x)
            xq = xq + eomt_obj.dp(ls_val)
            x, q = xq[:, eomt_obj.num_q :, :], xq[:, : eomt_obj.num_q, :]
            
            # --- Backbone Internal Logic (Applied Manually) ---
            
            # Self-Attention on X (Requires H, W)
            # x = x + drop_path(attn(norm1(x)))
            attn_out = block.attn(block.norm1(x), self.H, self.W)
            x = x + block.drop_path(attn_out)
            
            # MLP on X
            # x = x + drop_path(mlp(norm2(x)))
            x = x + block.drop_path(block.mlp(block.norm2(x)))
            
            # MLP on Q
            # q = q + drop_path(mlp(norm2(q)))
            q = q + block.drop_path(block.mlp(block.norm2(q)))
            
        else:
            # --- Standard ConvViT Block ---
            x = block(x, self.H, self.W)

        # 3. Reshape for next stage if necessary
        if (i+1) in self.depth_prefix_sum_list_post:
            x = x.reshape(self.B, self.H, self.W, self.C).permute(0, 3, 1, 2)
            
        return x, q

    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        # x is currently in NCHW format [B, C, H, W]
        # LayerNorm expects channels last [..., C]
        x = x.permute(0, 2, 3, 1)  # Change to [B, H, W, C]
        x = self.norm(x)
        x = x.permute(0, 3, 1, 2)  # Change back to [B, C, H, W]
        return x, q

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
