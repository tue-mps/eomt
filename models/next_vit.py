# ---------------------------------------------------------------
# © 2025 Mobile Perception Systems Lab at TU/e. All rights reserved.
# Licensed under the MIT License.
# ---------------------------------------------------------------

from typing import Optional
import timm
import torch
import torch.nn as nn
from models.backbones.next_vit import to_doubletuple
from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

def get_nextvit_final_grid(input_resolution):
    final_size = (input_resolution + 31) // 32
    return (final_size, final_size)


class NextViT(nn.Module):
    def __init__(
        self,
        img_size: tuple[int, int],
        backbone_name="nextvit_small_cus",
        multiplier: int = 1,
        ckpt_path: Optional[str] = None,
        **kwargs
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone_name,
            pretrained=ckpt_path is None,
            num_classes=0,       # Remove head
            img_size=img_size
        )

        # 2. Checkpoint Loading
        self.backbone = load_swin_ckpt_ignore_attn_mask(self.backbone, ckpt_path)
        # 3. Attributes for EoMT Compatibility
        
        self.patch_size = to_doubletuple(32) 
        self.grid_size = get_nextvit_final_grid(img_size[0])
        self.num_prefix_tokens = 0
        self.attn_multiplier = multiplier
        
        # Buffers
        pixel_mean = torch.tensor(IMAGENET_DEFAULT_MEAN).reshape(1, -1, 1, 1)
        pixel_std = torch.tensor(IMAGENET_DEFAULT_STD).reshape(1, -1, 1, 1)
        self.register_buffer("pixel_mean", pixel_mean)
        self.register_buffer("pixel_std", pixel_std)
        self.projection = None  # Will be defined in post_blocks when we know the final channel dimension
        self.q_projection = None
        self.blocks = nn.ModuleList()
        self.depths = self.backbone.depths
        self.num_heads = []
        self.token_norm = None
        for block in self.backbone.features:
            self.blocks.append(block)
            self.num_heads.append(block.mhca.groups)

        self.embed_dim = 768
        
        self.norm = self.backbone.norm
        #nn.LayerNorm(self.embed_dim)
        # State trackers
        self.B, self.C, self.H, self.W = 0, 0, 0, 0

    def pre_block(self, x: torch.Tensor):
        """
        Run the Stem: ConvBNReLU sequence
        """
        x = self.backbone.stem(x)
        self.B, self.C, self.H, self.W = x.shape
        return x

    
    def run_block(self, block, eomt_obj, x: torch.Tensor, q, i):
        """
        Unified block execution for NextViT with EoMT integration.
        Handles NCB/NTB blocks with spatial/token conversions internally.
        """
        
        # Track if we're in EoMT region
        in_eomt_region = i >= len(self.blocks) - eomt_obj.num_blocks
        
        if in_eomt_region:
            # Convert spatial to tokens if needed
            if x.dim() == 4:
                B, C, H, W = x.shape
                x = x.flatten(2).transpose(1, 2)  # (B, C, H, W) -> (B, N, C)
                self.grid_size = (H, W)
            
            # Concatenate queries with embeddings
            xq = torch.cat(
                (q[None, :, :].expand(x.shape[0], -1, -1), x), dim=1
            )
            
            # LayerNorm for token-space operations
            if not self.token_norm:
                self.token_norm = nn.LayerNorm(xq.shape[-1]).to(xq.device)
            
            pre_attn = self.token_norm(xq)
            x, q = pre_attn[:, eomt_obj.num_q:, :], pre_attn[:, :eomt_obj.num_q, :]
            
            # Prediction at this layer
            mask_logits, class_logits = eomt_obj._predict(x, q)
            eomt_obj.mask_logits_per_layer.append(mask_logits)
            eomt_obj.class_logits_per_layer.append(class_logits)
            
            # Cross-attention with EoMT module
            after_eomt = torch.cat((q, x), dim=1)
            new_x = eomt_obj.attn[i - len(self.blocks)](after_eomt)
            xq = xq + eomt_obj.dp(eomt_obj.ls_list[i - len(self.blocks)](new_x))
            x, q = xq[:, eomt_obj.num_q:, :], xq[:, :eomt_obj.num_q, :]
            
            # Reshape x back to spatial for standard block processing
            B, N, C = x.shape
            H, W = self.grid_size
            x_spatial = x.transpose(1, 2).reshape(B, C, H, W)
            
            # Run the block normally (BatchNorm2d applied inside on 4D tensors)
            x_spatial = block(x_spatial)
            
            # Update cache and convert back to tokens
            self.B, self.C, self.H, self.W = x_spatial.shape
            self.grid_size = (self.H, self.W)
            x = x_spatial
            
            # Process queries through MLP with dimension alignment
            q_in_channels = q.shape[-1]
            mlp_in_channels = block.mlp.conv1.in_channels
            
            # Project q if needed
            if q_in_channels != mlp_in_channels:
                if not hasattr(self, '_fallback_projs'):
                    self._fallback_projs = nn.ModuleDict()
                
                proj_key = f"proj_{q_in_channels}_{mlp_in_channels}"
                if proj_key not in self._fallback_projs:
                    new_proj = nn.Linear(q_in_channels, mlp_in_channels)
                    nn.init.kaiming_normal_(new_proj.weight)
                    nn.init.constant_(new_proj.bias, 0)
                    self._fallback_projs[proj_key] = new_proj
                
                # Ensure projection is on same device as q
                proj_layer = self._fallback_projs[proj_key]
                if proj_layer.weight.device != q.device:
                    proj_layer = proj_layer.to(q.device)
                    self._fallback_projs[proj_key] = proj_layer
            
                q = proj_layer(q)
            
            # Apply MLP to queries (reshape for Conv2d MLP)
            q_reshaped = q.transpose(1, 2).unsqueeze(-1)  # (B, N_q, C) -> (B, C, N_q, 1)

            q_reshaped = block.norm(q_reshaped) if hasattr(block, 'norm') else block.norm2(q_reshaped)
            
            q_out = block.mlp(q_reshaped)
            q_out = q_out.squeeze(-1).transpose(1, 2)  # (B, C, N_q, 1) -> (B, N_q, C)
            
            # Apply drop_path to query MLP output
            if hasattr(block, 'mlp_path_dropout'):
                q = q + block.mlp_path_dropout(q_out)
            else:
                q = q + q_out
            
        else:
            # Standard block forward (no EoMT)
            x = block(x)
            
            # Update spatial cache
            if x.dim() == 4:
                self.B, self.C, self.H, self.W = x.shape
                self.grid_size = (self.H, self.W)
        
        return x, q


    
    def post_blocks(self, x: torch.Tensor, q, eomt_obj):
        """
        Ensures the final output matches the expected embed_dim for the class heads.
        We project back to 768.
        """
        x = self.norm(x)
        # Determine current channel dimension
        current_dim = x.shape[1]
        if self.q_projection is None:
            self.q_projection = nn.Linear(current_dim, self.embed_dim).to(x.device)
            nn.init.kaiming_normal_(self.q_projection.weight)
            nn.init.constant_(self.q_projection.bias, 0)
        if self.projection is None:
            self.projection = nn.Linear(current_dim, self.embed_dim).to(x.device)
            nn.init.kaiming_normal_(self.projection.weight)
            nn.init.constant_(self.projection.bias, 0)
            # (B, C, H, W) -> (B, H, W, C) -> Linear -> (B, H, W, C_out) -> (B, C_out, H, W)
        x = x.permute(0, 2, 3, 1)
        x = self.projection(x)
        x = x.permute(0, 3, 1, 2)
        q = self.q_projection(q)
        return x,q


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

def pad2d_concat(b,a):
    P = a.shape[2]+b.shape[2]
    B, C = a.shape[:2]
    out = a.new_zeros(B, C, P, P)
    out[..., :a.shape[2], :a.shape[2]] = a
    out[..., a.shape[2]:P, a.shape[2]:P] = b
    return out

def pad2d_unconcat(out, a_size, b_size=None, clone=True):
    # out: [B,C,P,P], where P = a_size + b_size
    if b_size is None:
        b_size = out.shape[-1] - a_size  # assumes square P x P

    a = out[..., :a_size, :a_size]
    b = out[..., a_size:a_size+b_size, a_size:a_size+b_size]

    if clone:
        a = a.clone()
        b = b.clone()
    return b, a  # matches your original arg order (b,a)