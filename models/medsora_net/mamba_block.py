# Copyright (c) 2024, Zhenbin Wang (https://github.com/wongzbb/MedSora/).
# All rights reserved.

import torch
from einops import rearrange
from torch import nn

from timm.models.vision_transformer import Attention, Mlp
from local_attention import LocalAttention

from .SSM.spatio_mamba import Mamba2 as spatio_mamba
from .SSM.temporal_mamba import Mamba2 as temporal_mamba

def modulate(x, shift, scale):
    return x * (1 + scale.unsqueeze(1)) + shift.unsqueeze(1)

def modulate1(x, shift, scale, B, F, WH, D):
    x = x.view(B, F, WH*WH, D)
    x = x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    return x.view(B*F, WH*WH, D)

def modulate_gate(x, gate, B, F, WH, D):
    x = x.view(B, F, WH*WH, D)
    x = gate.unsqueeze(1).unsqueeze(1) * x
    return x.view(B*F, WH*WH, D)

def modulate2(x, shift, scale, B, F, WH, D):
    x = x.view(B, WH*WH, F, D)
    x = x * (1 + scale.unsqueeze(1).unsqueeze(1)) + shift.unsqueeze(1).unsqueeze(1)
    return x.view(B*WH*WH, F, D)

def modulate_f(x, gate, B, F, WH, D):
    x = x.view(B, WH*WH, F, D)
    x = gate.unsqueeze(1).unsqueeze(1) * x
    return x.view(B*WH*WH, F, D)

class MambaBlock(nn.Module): 
    def __init__(
        self,
        D_dim: int,
        E_dim: int,
        dt_rank: int,
        dim_inner: int,
        d_state: int,
        frame_token_list: list,
        frame_token_list_reversal: list,
        frame_origina_list: list,
        frame_origina_list_reversal: list,
        use_local_attention: bool,
    ):
        super().__init__()
        self.D_dim = D_dim
        self.E_dim = E_dim
        self.dt_rank = dt_rank
        self.frame_token_list = frame_token_list
        self.frame_token_list_reversal = frame_token_list_reversal
        self.frame_origina_list = frame_origina_list
        self.frame_origina_list_reversal = frame_origina_list_reversal
        self.use_local_attention = use_local_attention


        self.norm1 = nn.LayerNorm(D_dim)
        self.norm2 = nn.LayerNorm(D_dim)
        self.norm3 = nn.LayerNorm(D_dim)

        self.mamba_hw = spatio_mamba(
            d_model=D_dim, 
            d_state=d_state, 
            d_conv=4, 
            expand=2, 
            frame_token_list = self.frame_token_list,
            frame_token_list_reversal = self.frame_token_list_reversal,
            frame_origina_list = self.frame_origina_list,
            frame_origina_list_reversal = self.frame_origina_list_reversal,
            )
        self.mamba_f = temporal_mamba(
            d_model=D_dim, 
            d_state=d_state, 
            d_conv=4, 
            expand=2, 
            )

        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(D_dim, D_dim*9, bias=True),
        )


        if self.use_local_attention == True:
            # self.norm3 = nn.LayerNorm(D_dim)
            self.query_matrix = nn.Linear(D_dim, D_dim, bias=False)
            self.key_matrix = nn.Linear(D_dim, D_dim, bias=False)
            self.value_matrix = nn.Linear(D_dim, D_dim, bias=False)
            
            self.local_attention = LocalAttention(
                dim=D_dim, 
                window_size=8,  # window
                causal = False, # auto-regressive or not
                look_backward = 1, # each window looks at the window before
                look_forward = 1, # each window looks at itself
                dropout = 0.1, 
                exact_windowsize = True,
                )

        self.initialize_weights()

    def forward(self, x: torch.Tensor, c: torch.Tensor, WH: int, frame_space: torch.Tensor, use_local_cov: bool,):

        shift_msa, scale_msa, gate_msa, \
        shift_mamba, scale_mamba, gate_mamba, \
        shift_f, scale_f, gate_f = self.adaLN_modulation(c).chunk(9, dim=1)

        B, F, _, D = x.shape

        # start local attention
        if self.use_local_attention == True:
            x = rearrange(x, 'B F (H W) D -> (B F) (H W) D', F=F, H=WH, W=WH).contiguous()

            att_skip = x
            x = self.norm1(x)
            x = modulate1(x, shift_msa, scale_msa, B, F, WH, D)

            q_x = self.query_matrix(x)
            k_x = self.key_matrix(x)
            v_x = self.value_matrix(x)
            x = self.local_attention(q_x, k_x, v_x)

            x = att_skip + modulate_gate(x, gate_msa, B, F, WH, D)
        # end local attention
        
        
        # start spotio_mamba
        x = rearrange(x, '(B F) (H W) D -> B (F H W) D', B=B, H=WH, W=WH, D=D).contiguous()

        mamba_skip_hw = x
        x = self.norm2(x)
        x = modulate(x, shift_mamba, scale_mamba)

        x = self.mamba_hw(x, frame_num=F, WH=WH)
        x = rearrange(x, 'B F H W D -> B (F H W) D', B=B, H=WH, W=WH, D=D).contiguous()
        x = mamba_skip_hw + gate_mamba.unsqueeze(1) * x
        # end spotio_mamba
        
        if use_local_cov:  #for optical flow representation alignment
            local_x = rearrange(x, 'B (F H W) D -> B F (H W) D', F=F, H=WH, W=WH).contiguous()
        else:
            local_x = None

        # start temporal_mamba
        x = rearrange(x, 'B (F H W) D -> (B H W) F D', B=B, F=F, H=WH, W=WH).contiguous()
        mamba_skip_f = x
        x = self.norm3(x)

        x = modulate2(x, shift_f, scale_f, B, F, WH, D)
        x = self.mamba_f(x)
        x = mamba_skip_f + modulate_f(x, gate_f, B, F, WH, D)
        # end temporal_mamba

        x = rearrange(x, '(B H W) F D -> B F (H W) D',  B=B, F=F, H=WH, W=WH).contiguous()

        return x, local_x
    
    
    def initialize_weights(self):
        # Initialize parameter weights
        def _basic_init(module):
            if isinstance(module, nn.Linear):
                torch.nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
        self.apply(_basic_init)


        if self.use_local_attention == True:
            nn.init.constant_(self.query_matrix.weight, 0)
            nn.init.constant_(self.key_matrix.weight, 0)
            nn.init.constant_(self.value_matrix.weight, 0)
