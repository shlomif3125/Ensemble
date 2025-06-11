import pandas as pd
import torch
from torch import nn
import pytorch_lightning as pl
from .backbones import build_router_model 
from .metrics import (top_scoring_router_choiced_wer, 
                      thresholded_top_scoring_router_choiced_wer,
                      thresholded_weighted_router_choice_wer, 
                      f1_metric, recall_metric, precision_metric,
                      roc_auc_metric)


class MultiRouterModel(pl.LightningModule):
    def __init__(self, model_cfg):
        super(MultiRouterModel, self).__init__()
        self.cfg = model_cfg

        self.router_model = build_router_model(model_cfg.backbone)
        
        self.loss = nn.BCEWithLogitsLoss()   
        results_df_columns = ['tar_id', 'Model', 'RouterLabel','Score', 'instruction_type', 'w', 'we', 'baseline_we']
        self.validation_results_df = pd.DataFrame(columns=results_df_columns)
        self.test_results_df = pd.DataFrame(columns=results_df_columns)
        
        self.val_dl_names = model_cfg.val_dl_names
        self.val_dl_types = model_cfg.val_dl_types
    
    @staticmethod
    def calc_wer(w_we_df):
        return w_we_df['we'].sum() / w_we_df['w'].sum()

    def forward(self, x):
        out = self.router_model(x)
        return out
    
    def training_step(self, batch, batch_idx=0):
        _, x, router_labels, *_ = batch
        out = self.forward(x)
        y = torch.nn.functional.one_hot(router_labels.to(torch.int64), 2).to(torch.float32)
        loss = self.loss(out, y)
        self.log('loss', loss)
        return loss
    
    def on_validation_start(self):
        self.validation_results_df = self.validation_results_df.iloc[:0]
        self.test_results_df = self.test_results_df.iloc[:0]
        
    def validation_step(self, batch, batch_idx=0, dataloader_idx=0):
        tar_ids, x, router_labels, model_names, instruction_types, ws, wes, baseline_wes = batch
        out = self.forward(x)
        y = torch.nn.functional.one_hot(router_labels.to(torch.int64), 2).to(torch.float32)
        loss = self.loss(out, y)
        # loss = self.loss(out, router_labels.to(torch.float32))
        prefix = self.val_dl_types[dataloader_idx]
        self.log(f'{prefix}_loss/{self.val_dl_names[dataloader_idx]}', loss, add_dataloader_idx=False, batch_size=len(tar_ids), )
        
        router_labels = router_labels.cpu().detach().numpy()
        scores = nn.Softmax(dim=1)(out).cpu().detach().numpy()[:, 1]
        ws = ws.cpu().detach().numpy()
        wes = wes.cpu().detach().numpy()
        baseline_wes = baseline_wes.cpu().detach().numpy()
        batch_size = len(tar_ids)
        batch_rows = [dict(tar_id=tar_ids[i], 
                           Model=model_names[i], 
                           RouterLabel=router_labels[i], 
                           Score=scores[i],
                           instruction_type=instruction_types[i],
                           w=ws[i],
                           we=wes[i],
                           baseline_we=baseline_wes[i]) 
                      for i in range(batch_size)]
        batch_df = pd.DataFrame(batch_rows)
        if prefix == 'val':
            self.validation_results_df = pd.concat([self.validation_results_df, batch_df])
        elif prefix == 'test':
            self.test_results_df = pd.concat([self.test_results_df, batch_df])
        else:
            raise ValueError(f'No such validation-set type: {prefix}')

        return None
    
    #TODO: This waits for all validation-dataloader to finish, which is suboptimal. 
    def on_validation_epoch_end(self):
        self.validation_results_df = self.validation_results_df.reset_index(drop=True)
        for metric in [top_scoring_router_choiced_wer, 
                      thresholded_top_scoring_router_choiced_wer,
                      thresholded_weighted_router_choice_wer,
                      f1_metric, 
                      recall_metric, 
                      precision_metric,
                      roc_auc_metric
                      ]:
            
            if not self.validation_results_df.empty:
                instruction_type_wers, metric_name = metric(self.validation_results_df)
                if instruction_type_wers:
                    for k, v in instruction_type_wers.items():
                        self.log(f'val_{metric_name}/{k}', v)
                
            if not self.test_results_df.empty:
                instruction_type_wers, metric_name = metric(self.test_results_df)
                if instruction_type_wers:
                    for k, v in instruction_type_wers.items():
                        self.log(f'test_{metric_name}/{k}', v)        
    
    def configure_optimizers(self):
        weight_decay = self.cfg.optim.get('weight_decay', 0.0)
        opt = torch.optim.Adam(self.parameters(), lr=self.cfg.optim.lr, weight_decay=weight_decay)
        return opt

    
        