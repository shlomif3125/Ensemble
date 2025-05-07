from torch import nn
from .layers import SelfAttention, Transpose

def build_router_model(router_model_cfg):

    unsqueeze0 = nn.Unflatten(1, [1, router_model_cfg.embed_dim])
    conv0 = nn.Conv2d(1, 1, (5, 5), (2, 2))
    squeeze0 = nn.Flatten(1, 2)
    nl0 = nn.Mish()
    outdim0 = (router_model_cfg.embed_dim - 5) // 2 + 1
    
    conv1 = nn.Conv1d(outdim0, router_model_cfg.outdim1, 5, 2)
    nl1 = nn.Mish()
    norm1 = nn.BatchNorm1d(router_model_cfg.outdim1)
    sa1 = SelfAttention(router_model_cfg.outdim1, router_model_cfg.num_heads1, batch_first=True)
    
    conv2 = nn.Conv1d(router_model_cfg.outdim1, router_model_cfg.outdim2, 5, 2)
    nl2 = nn.Mish()
    norm2 = nn.BatchNorm1d(router_model_cfg.outdim2)
    sa2 = SelfAttention(router_model_cfg.outdim2, router_model_cfg.num_heads2, batch_first=True, )
    
    avg_pool = nn.AdaptiveAvgPool1d(router_model_cfg.avg_pool_out)
    conv3 = nn.Conv1d(router_model_cfg.outdim2, router_model_cfg.outdim3, router_model_cfg.avg_pool_out)
    squeeze = nn.Flatten(-2, -1)
    nl3 = nn.Mish()
    
    linear = nn.Linear(router_model_cfg.outdim3, router_model_cfg.out_dim)
    last_linear = nn.Linear(router_model_cfg.out_dim, 1)
    last_squeeze = nn.Flatten(0, -1)

    layer0 = nn.Sequential(unsqueeze0, conv0, squeeze0, nl0)
    layer1 = nn.Sequential(conv1, nl1, norm1, Transpose(1, 2), sa1, Transpose(1, 2))
    layer2 = nn.Sequential(conv2, nl2, norm2, Transpose(1, 2), sa2, Transpose(1, 2))
    layer_out = nn.Sequential(avg_pool, conv3, squeeze, nl3, linear, last_linear, last_squeeze)

    model = nn.Sequential(layer0, layer1, layer2, layer_out)
    return model