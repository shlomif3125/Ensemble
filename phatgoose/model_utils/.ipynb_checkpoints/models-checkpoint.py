import torch
from torch import nn
import pytorch_lightning as pl
from .layer_utils import SelfAttention, Transpose
    

class MultiRouterModelV0(pl.LightningModule):
    def __init__(self, model_names_list, 
                 embed_dim=512, 
                 outdim1=256, num_heads1=8, 
                 outdim2=64, num_heads2=8, 
                 avg_pool_out=3, outdim3=32,
                 out_dim=16):
        super(MultiRouterModelV0, self).__init__()

        module_dict = nn.ModuleDict({model_name: self.build_router_model(embed_dim=embed_dim, 
                                                                         outdim1=outdim1, num_heads1=num_heads1,
                                                                         outdim2=outdim2, num_heads2=num_heads2, 
                                                                         avg_pool_out=avg_pool_out, outdim3=outdim3,
                                                                         out_dim=out_dim) for model_name in model_names_list})
        self.module_dict = module_dict
        
        self.loss = nn.BCEWithLogitsLoss()        

        
    @staticmethod
    def build_router_model(embed_dim=512, 
                          outdim1=128, num_heads1=8,
                          outdim2=64, num_heads2=8, 
                          avg_pool_out=3, outdim3=32,
                          out_dim=16):

        unsqueeze0 = nn.Unflatten(1, [1, embed_dim])
        conv0 = nn.Conv2d(1, 1, (5, 5), (2, 2))
        squeeze0 = nn.Flatten(1, 2)
        nl0 = nn.Mish()
        outdim0 = (embed_dim - 5) // 2 + 1
        
        conv1 = nn.Conv1d(outdim0, outdim1, 5, 2)
        nl1 = nn.Mish()
        norm1 = nn.BatchNorm1d(outdim1)
        sa1 = SelfAttention(outdim1, num_heads1, batch_first=True)
        
        conv2 = nn.Conv1d(outdim1, outdim2, 5, 2)
        nl2 = nn.Mish()
        norm2 = nn.BatchNorm1d(outdim2)
        sa2 = SelfAttention(outdim2, num_heads2, batch_first=True, )
        
        avg_pool = nn.AdaptiveAvgPool1d(avg_pool_out)
        conv3 = nn.Conv1d(outdim2, outdim3, avg_pool_out)
        squeeze = nn.Flatten(-2, -1)
        nl3 = nn.Mish()
        
        linear = nn.Linear(outdim3, out_dim)
        last_linear = nn.Linear(out_dim, 1)
        last_squeeze = nn.Flatten(0, -1)


        layer0 = nn.Sequential(unsqueeze0, conv0, squeeze0, nl0)
        layer1 = nn.Sequential(conv1, nl1, norm1, Transpose(1, 2), sa1, Transpose(1, 2))
        layer2 = nn.Sequential(conv2, nl2, norm2, Transpose(1, 2), sa2, Transpose(1, 2))
        layer_out = nn.Sequential(avg_pool, conv3, squeeze, nl3, linear, last_linear, last_squeeze)

        model = nn.Sequential(layer0, layer1, layer2, layer_out)
        return model


    def forward(self, X, model_names):
        out = torch.cat([self.module_dict[mn][X[i:i+1]] for i, mn in enumerate(model_names)], dim=0)
        return out
    
    def training_step(self, batch, batch_idx=0):
        X, Y, model_names = batch
        out = self.forward(X, model_names)
        loss = self.loss(out, Y.to(torch.float32))
        return loss

    
        