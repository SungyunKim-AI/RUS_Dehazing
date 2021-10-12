import sys, os
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))

import torch
from torch import nn

from util.misc import *
from .transfomer import *
from .backbone import *

class VT(nn.Module):
    def __init__(self, num_queries, hidden_dim):
        super().__init__()
        self.num_queries = num_queries
        self.backbone = build_backbone(hidden_dim)
        self.transformer = Transformer(d_model=hidden_dim, nhead=8, num_encoder_layers=6,
                                      num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                                      activation="relu", normalize_before=False,
                                      return_intermediate_dec=False)
        
        self.query_embed = nn.Embedding(num_queries, hidden_dim)
        self.input_proj = nn.Conv2d(self.backbone.num_channels, hidden_dim, kernel_size=1)
        
        
    
    def forward(self, samples: NestedTensor):
        if isinstance(samples, (list, torch.Tensor)):
            samples = nested_tensor_from_tensor_list(samples)
        
        features, pos = self.backbone(samples)
        src, mask = features[-1].decompose()
        hs = self.transformer(self.input_proj(src), mask, self.query_embed.weight, pos[-1])[0]
        
        return hs
