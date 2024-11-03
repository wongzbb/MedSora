# Copyright (c) 2024, Zhenbin Wang.
import torch
import random
import torch.nn as nn
import numpy as np
import math
from timm.models.vision_transformer import Attention, Mlp #,PatchEmbed
from timm.models.layers import DropPath, to_2tuple
from functools import partial
from torch import Tensor
from typing import Optional
from einops import rearrange, repeat
from einops.layers.torch import Rearrange
from torch import nn, Tensor
from .mamba_block import MambaBlock
from tools import spiral
from torchmetrics.functional.regression import pearson_corrcoef

def modulate(x, shift, scale):
    if scale.shape[0]==1:
        return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)
    else:
        x = rearrange(x, '(b f) T D -> b f T D', b=scale.shape[0])    # (B, F, L, D)->(N, L, D)  
        x = x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
        x = rearrange(x, 'b f T D -> (b f) T D') 
        return x

#################################################################################
#                  Embedding Layers for Timesteps and Patch                     #
#################################################################################
class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    PatchEmbed(input_size, patch_size, in_channels, hidden_size, bias=True)
    """
    def __init__(self, img_size=16, patch_size=2, stride=2, in_chans=16, embed_dim=512, norm_layer=None, flatten=True):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = ((img_size[0] - patch_size[0]) // stride + 1, (img_size[1] - patch_size[1]) // stride + 1)
        self.num_patches = self.grid_size[0] * self.grid_size[1]
        self.flatten = flatten
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

    def forward(self, x):
        B, C, H, W = x.shape
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x)
        if self.flatten:
            x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        x = self.norm(x)
        return x


class TimestepEmbed(nn.Module):
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb



#################################################################################
#                                 Core MedSora model                            #
#################################################################################
class FinalLayer(nn.Module):
    """
    The final layer of MedSora.
    """
    def __init__(self, hidden_size, patch_size, out_channels):
        super().__init__()
        self.norm_final = nn.LayerNorm(hidden_size, elementwise_affine=False, eps=1e-6)
        self.linear = nn.Linear(hidden_size, patch_size * patch_size * out_channels, bias=True)
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(hidden_size, 2 * hidden_size, bias=True)
        )

    def forward(self, x, c):
        shift, scale = self.adaLN_modulation(c).chunk(2, dim=1)
        x = modulate(self.norm_final(x), shift, scale)
        x = self.linear(x)
        return x


class MedSora(nn.Module):
    """
    Diffusion model with a Mamba backbone.
    """
    def __init__(
        self,
        input_size=16,
        patch_size=2,
        strip_size = 2,
        in_channels=16,
        hidden_size=512,
        depth=16,
        learn_sigma=True,
        dt_rank=32,
        d_state=32,
        latent_num_all=6,
        latent_num_frames=4,
        use_image_num=8,
        use_covariance=True,
        what2Video='N2V',
        use_local_attention=False,
        use_mamba2=False,
        use_local_cov=False,
    ):
        super().__init__()
        self.learn_sigma = learn_sigma
        self.depth = depth
        self.in_channels = in_channels
        self.out_channels = in_channels * 2 if learn_sigma else in_channels
        self.patch_size = patch_size
        self.input_size = input_size
        self.use_image_num = use_image_num
        self.x_embedder = PatchEmbed(input_size, patch_size, strip_size, in_channels, hidden_size)
        self.t_embedder = TimestepEmbed(hidden_size)
        self.use_covariance = use_covariance
        self.what2Video = what2Video
        self.use_local_attention=use_local_attention
        self.latent_num_frames=latent_num_frames
        self.latent_num_all=latent_num_all
        self.use_local_cov = use_local_cov  ## use local covariance


        num_patches = self.x_embedder.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, hidden_size), requires_grad=False)

        frame_order_list, frame_original_order_indexes_list = spiral(self.latent_num_all, 
                                                          int(self.input_size/self.patch_size))

        self.blocks = nn.ModuleList([
            MambaBlock(frame_token_list=frame_order_list[(2*i)%len(frame_order_list)], 
                        frame_token_list_reversal=frame_order_list[(2*i)%len(frame_order_list)+1], 
                        frame_origina_list=frame_original_order_indexes_list[(2*i)%len(frame_order_list)], 
                        frame_origina_list_reversal=frame_original_order_indexes_list[(2*i)%len(frame_order_list)+1], 
                        D_dim=hidden_size, 
                        E_dim=hidden_size*2, 
                        dim_inner=hidden_size*2,
                        dt_rank=dt_rank,  
                        d_state=d_state,
                        use_local_attention=self.use_local_attention,) 
            for i in range(depth) 
        ])

        self.final_layer = FinalLayer(hidden_size, patch_size, self.out_channels)

        if self.what2Video == 'I2V':
            self.y_embedder = nn.Sequential(
                nn.SiLU(),
                nn.Linear(self.in_channels*self.input_size**2, hidden_size),
            )
            

        self.initialize_weights()

    def initialize_weights(self):
        # Initialize mamba layers:
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)

        # Initialize (and freeze) pos_embed by sin-cos embedding:
        pos_embed = get_2d_sincos_pos_embed(self.pos_embed.shape[-1], int(self.x_embedder.num_patches ** 0.5))
        self.pos_embed.data.copy_(torch.from_numpy(pos_embed).float().unsqueeze(0))

        # Initialize patch_embed like nn.Linear (instead of nn.Conv2d):
        w = self.x_embedder.proj.weight.data
        nn.init.xavier_uniform_(w.view([w.shape[0], -1]))
        nn.init.constant_(self.x_embedder.proj.bias, 0)

        nn.init.normal_(self.t_embedder.mlp[0].weight, std=0.02)
        nn.init.normal_(self.t_embedder.mlp[2].weight, std=0.02)

        # Zero-out adaLN modulation layers in mamba blocks:
        for block in self.blocks:
            nn.init.constant_(block.adaLN_modulation[-1].weight, 0)
            nn.init.constant_(block.adaLN_modulation[-1].bias, 0)

        # Zero-out output layers:
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].weight, 0)
        nn.init.constant_(self.final_layer.adaLN_modulation[-1].bias, 0)
        nn.init.constant_(self.final_layer.linear.weight, 0)
        nn.init.constant_(self.final_layer.linear.bias, 0)

        if self.what2Video == 'I2V':
            nn.init.constant_(self.y_embedder[-1].weight, 0)
            nn.init.constant_(self.y_embedder[-1].bias, 0)

    def unpatchify(self, x):
        """
        x: (N, T, patch_size**2 * C)
        imgs: (N, H, W, C)
        """
        c = self.out_channels
        p = self.x_embedder.patch_size[0]
        h = w = int(x.shape[1] ** 0.5)
        assert h * w == x.shape[1]

        x = x.reshape(shape=(x.shape[0], h, w, p, p, c))
        x = torch.einsum('nhwpqc->nchpwq', x)
        imgs = x.reshape(shape=(x.shape[0], c, h * p, h * p))
        return imgs

    def forward(self, x, t, frame_space):
        b, f, c, h, w = x.shape
        x = rearrange(x, 'b f c h w -> (b f) c h w', b=b).contiguous()
        x = self.x_embedder(x) + self.pos_embed   
        x = rearrange(x, '(b f) T D -> b f T D', b=b).contiguous()  

        # Get timestep embeddings:
        t = self.t_embedder(t)   

        # Image2Video, Text2Video, or None2Video
        # if what2video == 'I2V':
        #     y = self.y_embedder(y) # [B, D]  [4, 512]
        #     c = t + y
        # elif what2video == 'T2V':
        #     # coming soon.....
        #     c = t
        # else:
        c = t

        # Forward pass through the blocks:
        block_attention_list = []
        local_attention_list = []
        for block in self.blocks:
            x, local_x = block(x=x, 
                      c=c, 
                      WH=int(self.input_size/self.patch_size), 
                      frame_space=frame_space, 
                      use_local_cov=self.use_local_cov,
                      )
            block_attention_list.append(x[:,0:self.latent_num_frames,:,:])
            if local_x is not None:
                local_attention_list.append(local_x[:,0:self.latent_num_frames,:,:])

        b, f, T, D = x.shape
        x = rearrange(x, 'b f T D -> (b f) T D', b=b) 
        x = self.final_layer(x, c)      
        x = self.unpatchify(x) 

        x = rearrange(x, '(b f) c h w -> b f c h w', b=b)

        return x, frame_space, block_attention_list, local_attention_list, self.use_covariance, self.use_local_attention

    def forward_with_cfg(self, x, t, frame_space, cfg_scale):
        """
        Forward pass of MedSora, but also batches the unconditional forward pass for classifier-free guidance.
        """
        half = x[: len(x) // 2]
        combined = torch.cat([half, half], dim=0)
        model_out = self.forward(combined, t, frame_space)
        eps, rest = model_out[:, :3], model_out[:, 3:]
        cond_eps, uncond_eps = torch.split(eps, len(eps) // 2, dim=0)
        half_eps = uncond_eps + cfg_scale * (cond_eps - uncond_eps)
        eps = torch.cat([half_eps, half_eps], dim=0)
        return torch.cat([eps, rest], dim=1)

#################################################################################
#                   Sine/Cosine Positional Embedding Functions                  #
#################################################################################
# from https://github.com/facebookresearch/mae/blob/main/util/pos_embed.py

def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)
    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


#################################################################################
#                                   MedSora Configs                             #
#################################################################################

def MedSora_XL(**kwargs):
    return MedSora(depth=48, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

def MedSora_L(**kwargs):
    return MedSora(depth=24, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

def MedSora_B(**kwargs):
    return MedSora(depth=12, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

def MedSora_S(**kwargs):
    return MedSora(depth=6, hidden_size=512, patch_size=2, strip_size=2, **kwargs)

# def MedSora_XL_4(**kwargs):
#     return MedSora(depth=48, hidden_size=512, patch_size=4, strip_size=4, **kwargs)
# def MedSora_XL_7(**kwargs): 
#     return MedSora(depth=48, hidden_size=512, patch_size=8, strip_size=8, **kwargs)
# def MedSora_L_4(**kwargs):
#     return MedSora(depth=24, hidden_size=512, patch_size=4, strip_size=4, **kwargs)
# def MedSora_L_7(**kwargs):  
#     return MedSora(depth=24, hidden_size=512, patch_size=7, strip_size=8, **kwargs)
# def MedSora_B_4(**kwargs):
#     return MedSora(depth=12, hidden_size=512, patch_size=4, strip_size=4, **kwargs)
# def MedSora_B_7(**kwargs): 
#     return MedSora(depth=12, hidden_size=512, patch_size=7, strip_size=8, **kwargs)
# def MedSora_S_4(**kwargs):
#     return MedSora(depth=6, hidden_size=512, patch_size=4, strip_size=4, **kwargs)
# def MedSora_S_7(**kwargs): 
#     return MedSora(depth=6, hidden_size=512, patch_size=7, strip_size=8, **kwargs)


MedSora_models = {
    #---------------------------------------Ours------------------------------------------#
    'MedSora-XL': MedSora_XL,  #'MedSora-XL/4': MedSora_XL_4,  'MedSora-XL/7': MedSora_XL_7,
    'MedSora-L' : MedSora_L,   #'MedSora-L/4' : MedSora_L_4,   'MedSora-L/7' : MedSora_L_7,
    'MedSora-B' : MedSora_B,   #'MedSora-B/4' : MedSora_B_4,   'MedSora-B/7' : MedSora_B_7,
    'MedSora-S' : MedSora_S,   #'MedSora-S/4' : MedSora_S_4,   'MedSora-S/7' : MedSora_S_7,
}
