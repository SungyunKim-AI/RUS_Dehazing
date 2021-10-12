import warnings
warnings.filterwarnings('ignore')

import torch
from torch import nn
import models
from util.misc import nested_tensor_from_tensor_list

if __name__=='__main__':
    samples = torch.randn((16,3,256,256))
    # Backbone
    hidden_dim = 256
    backbone_num_channels = 2048     # 512 if name in ('resnet18', 'resnet34')
    input_proj = nn.Conv2d(backbone_num_channels, hidden_dim, kernel_size=1)
    backbone = models.build_backbone(hidden_dim, masks=True, lr_backbone=1e-5, dilation=False)
    
    num_queries = 100
    query_embed = nn.Embedding(num_queries, hidden_dim)
    transformer = models.Transformer(d_model=256, nhead=8, num_encoder_layers=6,
                                     num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                                     activation="relu", normalize_before=False,
                                     return_intermediate_dec=False)
    
    if isinstance(samples, (list, torch.Tensor)):
        samples = nested_tensor_from_tensor_list(samples)
    features, pos = backbone(samples)

    src, mask = features[-1].decompose()
    
    hs = transformer(input_proj(src), mask, query_embed.weight, pos[-1])[0]
    print("Output Shape : ", hs.shape)