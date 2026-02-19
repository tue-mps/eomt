

from typing import Optional

import timm
import torch
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn as nn
from models.backbones.convViT import to_doubletuple

class BackboneNorm(nn.Module):
    def __init__(self, norm_layer: nn.Module, grid_size: Optional[tuple] = None):
        super().__init__()
        self.norm = norm_layer
        self.grid_size = grid_size # (H, W) tuple to help reshape tokens if needed

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Case 1: Standard Spatial (B, C, H, W)
        if x.dim() == 4:
            return self.norm(x)
            
        # Case 2: Token sequence (B, N, C)
        # BatchNorm2d expects (B, C, Spatial). We can treat N as H*W or just W=1.
        if x.dim() == 3:
            B, N, C = x.shape
            # Permute to (B, C, N) -> reshape to (B, C, N, 1) for BN2d
            x_perm = x.transpose(1, 2).unsqueeze(-1) 
            x_norm = self.norm(x_perm)
            # Permute back: (B, C, N, 1) -> (B, C, N) -> (B, N, C)
            return x_norm.squeeze(-1).transpose(1, 2)
            
        return x

def get_efficientvit_final_grid(input_resolution):
    final_size = (input_resolution + 31) // 32
    return (final_size, final_size)

class EfficientNet(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="efficient_vit_b2", 
        multiplier: int = 1,
        ckpt_path: Optional[str] = None
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            img_size=img_size,
            num_classes=0
        )
        self.backbone = load_swin_ckpt_ignore_attn_mask(self.backbone, ckpt_path).backbone
        self.embed_dim = self.backbone.width_list[-1]
        self.patch_size = to_doubletuple(32) 
        self.grid_size = get_efficientvit_final_grid(img_size[0])
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier
        self.blocks = nn.ModuleList()
        self.depths = []
        self.num_heads = [] 
        
        self.patch_embed_dict = nn.ModuleDict()
        self.patch_embed_dict['0'] = self.backbone.input_stem
        
        for stage_idx, stage in enumerate(self.backbone.stages):
            stage_depth = len(stage)
            self.depths.append(stage_depth)
            # Assign dummy heads (EfficientViT doesn't expose this externally per block)
            stage_heads = 8 if stage_idx >= 2 else 1
            for block in stage:
                self.blocks.append(block)
                self.num_heads.append(stage_heads)
        if hasattr(self.backbone, 'norm') and self.backbone.norm is not None:
            raw_norm = self.backbone.norm
        else:
            # Default to BatchNorm2d matching final embedding dim
            # This mimics the standard behavior of EfficientNet/MobileNet final norms
            raw_norm = nn.BatchNorm2d(self.embed_dim, eps=1e-5)
            
        self.norm = BackboneNorm(raw_norm, grid_size=self.grid_size)
        
        self.B, self.C, self.H, self.W = 0, 0, 0, 0
        self.attn_multiplier = multiplier
        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(
            1, -1, 1, 1
        )
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
    
    def pre_block(self, x: torch.Tensor):
        # Run stem (index '0' in our dict)
        if '0' in self.patch_embed_dict:
            x = self.patch_embed_dict['0'](x)
            self.B, self.C, self.H, self.W = x.shape
        return x
    
    def q_mlp(self, block, q: torch.Tensor):
        """
        Apply the MLP (Feed-Forward) parts of the backbone block to the queries.
        For EfficientViT, this means running the 1x1 convolutions (Expansion & Projection)
        of the MBConv block, while skipping the spatial Depthwise Convolution.
        """
        # 1. Unwrap the block to find the core MBConv
        # efficient_vit_block (LiteMLA+MBConv) -> local_module (Residual) -> MBConv
        # residual_block -> main (MBConv)
        
        target_module = block
        use_residual = False
        
        # Check if it's a ResidualBlock (most blocks in the stages are)
        if hasattr(block, 'main') and hasattr(block, 'shortcut'):
            # If shortcut is Identity or None, we should add residual at the end
            # (Assuming standard ResNet layout: y = f(x) + x)
            is_identity = isinstance(block.shortcut, (nn.Identity, type(None))) or \
                          (isinstance(block.shortcut, nn.Sequential) and len(block.shortcut)==0)
            if is_identity:
                use_residual = True
            
            target_module = block.main # Unwrap
            
        # Check if it's an EfficientViTBlock (contains context_module and local_module)
        if hasattr(target_module, 'local_module'):
            # The FFN part is in the local_module
            return self.q_mlp(target_module.local_module, q)
            
        # 2. Apply MBConv Logic (Channel Mixing only)
        # We look for 'inverted_conv' and 'point_conv'
        if hasattr(target_module, 'inverted_conv') and hasattr(target_module, 'point_conv'):
            
            # Prepare Queries: (B, N, C) -> (B, C, N, 1) to use Conv2d layers
            B, N, C = q.shape
            q_reshaped = q.transpose(1, 2).unsqueeze(-1)
            
            # A. Expansion (1x1 Conv + BN + Act)
            # This is equivalent to the first Linear layer in a standard ViT FFN
            x = target_module.inverted_conv(q_reshaped)
            
            # B. Depthwise (Spatial) - SKIPPED
            # Queries have no spatial arrangement, so we treat this as Identity.
            # Note: We effectively skip the 'Act' inside depth_conv, but inverted_conv has Act.
            
            # C. Projection (1x1 Conv + BN)
            # This is equivalent to the second Linear layer
            x = target_module.point_conv(x)
            
            # Reshape back: (B, C, N, 1) -> (B, N, C)
            q_out = x.squeeze(-1).transpose(1, 2)
            
            # 3. Residual Connection
            if use_residual:
                q_out = q_out + q
                
            return q_out

        # Fallback: If block structure is unknown or not FFN-like, return q unchanged
        return q
    
    def run_block(self, block, x: torch.Tensor, curr: int):
        # Case 1: Standard Spatial (B, C, H, W)
        if x.dim() == 4:
            x = block(x)
            self.B, self.C, self.H, self.W = x.shape
            return x
            
        # Case 2: Token Input (B, N, C) - from EoMT loop
        elif x.dim() == 3:
            B, N, C = x.shape
            
            H, W = self.grid_size
            
            # Use cached H,W if grid_size is static or mismatch
            if H * W != N and self.H * self.W == N:
                H, W = self.H, self.W
            
            x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
            x_spatial = block(x_spatial)
            
            # Update cache
            self.B, self.C, self.H, self.W = x_spatial.shape
            self.grid_size = (self.H, self.W)
            
            # Flatten back
            x_tokens = x_spatial.flatten(2).transpose(1, 2)
            return x_tokens
            
        return x
    
def load_swin_ckpt_ignore_attn_mask(model, ckpt_path):
    # 1) Load checkpoint state dict
    checkpoint = torch.load(ckpt_path, map_location="cpu",weights_only=False)
    if 'model_state' in checkpoint:
        state_dict = {k[7:] if k.startswith('module.') else k: v for k, v in checkpoint['model_state'].items()}
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]


    # 2) Build a reference of current model shapes for conditional filtering
    model_state = model.state_dict()


    # 3) Load with strict=False to tolerate any remaining non-critical diffs
    missing, unexpected = model.load_state_dict(model_state, strict=False)
    print("Missing keys:", missing)
    print("Unexpected keys:", unexpected)
    return model