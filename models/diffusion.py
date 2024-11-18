import torch
import torch.nn as nn
import einops
from typing import Optional, Tuple, List, Dict
import numpy as np
import math

class DiffusionModel(nn.Module):
    """Diffusion model for synthetic sample generation."""
    
    def __init__(
        self,
        img_size: int = 64,
        in_channels: int = 4,
        model_channels: int = 1536,
        num_heads: int = 24,
        text_dim: int = 64,
        num_text_tokens: int = 77,
        clip_img_dim: int = 512,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.img_size = img_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        
        # Define beta schedule for diffusion
        self.register_buffer('betas', self._get_beta_schedule())
        alphas = 1. - self.betas
        self.register_buffer('alphas_cumprod', torch.cumprod(alphas, dim=0))
        
        # Image processing
        self.image_proj = nn.Sequential(
            nn.Conv2d(in_channels, model_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(model_channels, model_channels, 3, padding=1)
        )
        
        # Text processing
        self.text_proj = nn.Sequential(
            nn.Linear(text_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
        
        # CLIP image processing
        self.clip_proj = nn.Sequential(
            nn.Linear(clip_img_dim, model_channels),
            nn.SiLU(),
            nn.Linear(model_channels, model_channels)
        )
        
        # Time embedding
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_dim),
            nn.SiLU(),
            nn.Linear(time_embed_dim, time_embed_dim),
        )
        
        # Main transformer
        self.transformer = TransformerModel(
            dim=model_channels,
            depth=30,
            heads=num_heads,
            dim_head=64,
            mlp_dim=model_channels * 4,
            use_checkpoint=use_checkpoint
        )
        
        # Output projections
        self.z_proj = nn.Sequential(
            nn.Conv2d(model_channels, in_channels, 3, padding=1),
            nn.SiLU(),
            nn.Conv2d(in_channels, in_channels, 3, padding=1)
        )
        
        self.clip_img_proj = nn.Sequential(
            nn.Linear(model_channels, clip_img_dim),
            nn.SiLU(),
            nn.Linear(clip_img_dim, clip_img_dim)
        )
        
        self.text_out_proj = nn.Sequential(
            nn.Linear(model_channels, text_dim),
            nn.SiLU(),
            nn.Linear(text_dim, text_dim)
        )

    def _get_beta_schedule(self, schedule_type='linear'):
        """Get beta schedule for diffusion process."""
        if schedule_type == 'linear':
            scale = 1000 / 999
            beta_start = scale * 0.0001
            beta_end = scale * 0.02
            return torch.linspace(beta_start, beta_end, 1000, dtype=torch.float32)
        else:
            raise NotImplementedError(f"Unknown schedule type {schedule_type}")
            
    def q_sample(self, x_start: torch.Tensor, t: torch.Tensor, noise: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward diffusion process."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        alphas_cumprod_t = self.alphas_cumprod.index_select(0, t)
        alphas_cumprod_t = alphas_cumprod_t.view(-1, 1, 1, 1)
        
        x_t = torch.sqrt(alphas_cumprod_t) * x_start + \
              torch.sqrt(1 - alphas_cumprod_t) * noise
        return x_t
        
    def p_losses(
        self, 
        x_start: torch.Tensor,
        t: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        clip_img: Optional[torch.Tensor] = None,
        noise: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Calculate diffusion losses."""
        if noise is None:
            noise = torch.randn_like(x_start)
            
        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted = self(x_noisy, t, text, clip_img)
        
        loss = torch.nn.functional.mse_loss(predicted, noise)
        return loss
        
    def forward(
        self,
        x: torch.Tensor,
        timesteps: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        clip_img: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass."""
        # Embed time
        temb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        
        # Process inputs
        x = self.image_proj(x)
        if text is not None:
            text = self.text_proj(text)
        if clip_img is not None:
            clip_img = self.clip_proj(clip_img)
            
        # Combine features
        if text is not None and clip_img is not None:
            context = torch.cat([text, clip_img], dim=1)
        elif text is not None:
            context = text
        elif clip_img is not None:
            context = clip_img
        else:
            context = None
            
        # Apply transformer
        h = self.transformer(x, context, temb)
        
        # Generate outputs
        z_out = self.z_proj(h)
        clip_img_out = self.clip_img_proj(h.mean(dim=[2, 3]))
        text_out = self.text_out_proj(h.mean(dim=[2, 3]))
        
        return z_out, clip_img_out, text_out
        
    @torch.no_grad()
    def p_sample(
        self,
        x: torch.Tensor,
        t: torch.Tensor,
        text: Optional[torch.Tensor] = None,
        clip_img: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Sample from the model."""
        b = x.shape[0]
        
        # Predict noise
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x,
            t=t,
            text=text,
            clip_img=clip_img
        )
        
        noise = torch.randn_like(x)
        nonzero_mask = (t != 0).float().view(-1, 1, 1, 1)
        
        # Sample
        return model_mean + nonzero_mask * torch.exp(0.5 * model_log_variance) * noise
        
    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        text: Optional[torch.Tensor] = None,
        clip_img: Optional[torch.Tensor] = None,
        device: Optional[torch.device] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Generate samples."""
        device = device or next(self.parameters()).device
        
        shape = (batch_size, self.in_channels, self.img_size, self.img_size)
        img = torch.randn(shape, device=device)
        
        for t in reversed(range(0, len(self.betas))):
            img = self.p_sample(
                x=img,
                t=torch.full((batch_size,), t, device=device, dtype=torch.long),
                text=text,
                clip_img=clip_img
            )
            
        return img

class TransformerModel(nn.Module):
    """Transformer model for feature processing."""
    
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int,
        mlp_dim: int,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.layers = nn.ModuleList([])
        self.use_checkpoint = use_checkpoint
        
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head)),
                PreNorm(dim, FeedForward(dim, mlp_dim))
            ]))
            
    def forward(
        self, 
        x: torch.Tensor,
        context: Optional[torch.Tensor] = None,
        time_emb: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass."""
        
        def _inner_forward(x: torch.Tensor, attn: nn.Module, ff: nn.Module) -> torch.Tensor:
            x = attn(x, context=context) + x
            x = ff(x + time_emb) + x
            return x
            
        for attn, ff in self.layers:
            if self.use_checkpoint and x.requires_grad:
                x = checkpoint.checkpoint(_inner_forward, x, attn, ff)
            else:
                x = _inner_forward(x, attn, ff)
                
        return x

def timestep_embedding(timesteps: torch.Tensor, dim: int, max_period: int = 10000) -> torch.Tensor:
    """Create sinusoidal timestep embeddings."""
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding