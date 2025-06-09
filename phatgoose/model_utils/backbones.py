from torch import nn
from .layers import SelfAttention, Transpose
from torchvision.models.resnet import resnet18

in1 = 512
out1 = in2 = 256
out2 = in3 = 672



def build_router_model(router_model_cfg):
    
    if router_model_cfg.type == 'attention':
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
    
    elif router_model_cfg.type == 'resnet':
        time_norm = nn.InstanceNorm1d(router_model_cfg.embed_dim)
        conv1 = nn.Conv1d(router_model_cfg.embed_dim, router_model_cfg.outdim1, 3, 1)
        nl1 = nn.LeakyReLU()
        bn1 = nn.BatchNorm1d(router_model_cfg.outdim1)
        conv2 = nn.Conv1d(router_model_cfg.outdim1, router_model_cfg.outdim2, 5, 2)
        nl2 = nn.LeakyReLU()
        bn2 = nn.BatchNorm1d(router_model_cfg.outdim2)
        avg_pool = nn.AdaptiveAvgPool1d(router_model_cfg.outdim2)
        nl3 = nn.LeakyReLU()
        unflatten = nn.Unflatten(1, [3, 224])
        res18 = resnet18()
        res18.fc = nn.Linear(512, 4)
        nl4 = nn.Mish()
        linear = nn.Linear(4, 2)
        # linear = nn.Linear(4, 1)
        # flatten = nn.Flatten(0)
        model = nn.Sequential(time_norm,
                              conv1, nl1, bn1,
                              conv2, nl2, bn2,
                              avg_pool, nl3, unflatten, 
                              res18, nl4,
                              linear)#, flatten)
    
    elif router_model_cfg.type == 'attention_inner_layers':
        outdim = router_model_cfg.outdim1
        num_heads = router_model_cfg.num_heads1
        transpose_layer = Transpose(1, 2)
        num_self_attention_layers = router_model_cfg.num_self_attention_layers
        stem = nn.Sequential(nn.Conv1d(1024, outdim, 1), nn.LeakyReLU(), transpose_layer)
        model_layers = [stem]
        for _ in range(num_self_attention_layers):
            layer = nn.Sequential(SelfAttention(outdim, num_heads, batch_first=True), 
                                nn.Mish(), 
                                transpose_layer,
                                nn.Conv1d(outdim, outdim, 5, 2, groups=outdim),
                                nn.LeakyReLU(),
                                transpose_layer)
            model_layers.append(layer)

        out_layer = nn.Sequential(transpose_layer, nn.AdaptiveAvgPool1d(1), nn.Flatten(), nn.Linear(outdim, 64), nn.LeakyReLU(), nn.Linear(64, 2))
        model_layers.append(out_layer)
        model = nn.Sequential(*model_layers)
            
    elif router_model_cfg.type == 'resnet_inner_layers':
        # time_norm = nn.InstanceNorm1d(router_model_cfg.embed_dim)
        
        # transpose = Transpose(1, 2)
        conv1 = nn.Conv1d(router_model_cfg.embed_dim, router_model_cfg.outdim1, 3, 1)
        nl1 = nn.LeakyReLU()
        bn1 = nn.BatchNorm1d(router_model_cfg.outdim1)
        avg_pool = nn.AdaptiveAvgPool1d(router_model_cfg.outdim2)
        nl2 = nn.LeakyReLU()
        unflatten = nn.Unflatten(1, [3, 112, 2])
        flatten = nn.Flatten(-2)
        upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        res18 = resnet18()
        res18.fc = nn.Linear(512, 4)
        nl3 = nn.Mish()
        linear = nn.Linear(4, 2)
        # linear = nn.Linear(4, 1)
        # flatten = nn.Flatten(0)
        model = nn.Sequential(#time_norm,
                            #   transpose,
                              conv1, nl1, bn1,
                              avg_pool, nl2, 
                              unflatten, flatten, upsample,
                              res18, nl3,
                              linear)#, flatten)
    else:
        raise NotImplementedError(f"No known backbone implementation for type {router_model_cfg.type}")
    
    return model

    