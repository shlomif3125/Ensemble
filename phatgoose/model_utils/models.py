import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from .backbones import build_router_model    

class MultiRouterModel(pl.LightningModule):
    def __init__(self, model_cfg):
        super(MultiRouterModel, self).__init__()
        self.cfg = model_cfg

        module_dict = nn.ModuleDict({model_name: build_router_model(model_cfg.backbone) 
                                     for model_name in model_cfg.model_names_list})
        self.module_dict = module_dict
        
        self.loss = nn.BCEWithLogitsLoss()   
        self.validation_results_df = pd.DataFrame(columns=['tar_id',
                                                           'Model', 
                                                           'RouterLabel',
                                                           'Score', 
                                                           'instruction_type', 
                                                           'w',
                                                           'we'])

    def forward(self, x, model_names):
        if len(set(model_names)) == 1:
            m = self.module_dict[model_names[0]]
            out = m(x)
        else:
            # TODO: group by model-name, and restore original order afterwards, to gain mini-batches utilization
            out = torch.cat([self.module_dict[mn](x[i:i+1]) for i, mn in enumerate(model_names)], dim=0)
        return out
    
    def training_step(self, batch, batch_idx=0):
        _, x, router_labels, model_names, *_ = batch
        out = self.forward(x, model_names)
        loss = self.loss(out, router_labels.to(torch.float32))
        self.log('loss', loss)
        return loss
    
    def on_validation_start(self):
        self.validation_results_df = self.validation_results_df.iloc[:0]
        
    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        tar_ids, x, router_labels, model_names, instruction_types, ws, wes = batch
        out = self.forward(x, model_names)
        loss = self.loss(out, router_labels.to(torch.float32))
        self.log('val_loss', loss, add_dataloader_idx=False)
        
        router_labels = router_labels.cpu().detach().numpy()
        scores = nn.Sigmoid()(out).cpu().detach().numpy()
        ws = ws.cpu().detach().numpy()
        wes = wes.cpu().detach().numpy()
        batch_size = len(tar_ids)
        batch_rows = [dict(tar_id=tar_ids[i], 
                           Model=model_names[i], 
                           RouterLabel=router_labels[i], 
                           Score=scores[i],
                           instruction_type=instruction_types[i],
                           w=ws[i],
                           we=wes[i]) 
                      for i in range(batch_size)]
        batch_val_df = pd.DataFrame(batch_rows)
        self.validation_results_df = pd.concat([self.validation_results_df, batch_val_df])
        
        return loss

    @staticmethod
    def calc_wer(w_we_df):
        return w_we_df['we'].sum() / w_we_df['w'].sum()
    
    #TODO: This waits for all validation-dataloader to finish, which is suboptimal. 
    def on_validation_epoch_end(self):
        self.validation_results_df = self.validation_results_df.reset_index(drop=True)
        validation_results_df = self.validation_results_df
        best_model_per_tar_id_df = validation_results_df.loc[validation_results_df.groupby('tar_id')['Score'].idxmax(), ['instruction_type', 'w', 'we']]
        instruction_type_wers = best_model_per_tar_id_df.groupby('instruction_type').apply(self.calc_wer).to_dict()
        for k, v in instruction_type_wers.items():
            self.log(f'val_{k}_wer_by_chosen_model', v)
        
    
    def configure_optimizers(self):
        opt = torch.optim.Adam(self.parameters(), self.cfg.optim.lr)
        return opt

    
        