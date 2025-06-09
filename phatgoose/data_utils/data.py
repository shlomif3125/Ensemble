import random
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
import logging


class MultiRouterDataset(Dataset):
    def __init__(self, data_cfg):
        self.cfg = data_cfg
        logging.info('Loading Manifest File')
        self.dataset_df = pd.read_pickle(self.cfg.manifest_filepath)
        
        logging.info(f'Filtering dataset for model names {self.cfg.get("model_names", None)}')
        self.filter_dataset_df()
        
        logging.info('Processing Dataset')
        self.process_dataset_df()
        
        self.model_names_list = sorted(set(self.dataset_df['Model']))
        logging.info(f'END OF INIT model_names_list: {self.model_names_list}')
        
    def filter_dataset_df(self):
        model_names = self.cfg.get('model_names', None)

        if model_names is not None:
            if type(model_names) == str:
                model_names = [model_names]
            model_names.append(self.cfg.baseline_model_name)
            self.dataset_df = self.dataset_df[self.dataset_df['Model'].isin(model_names)]
            
    def unify_models(self, df):
        print('UNIFYING MODELS')
        unified_model_name = '__'.join(sorted(set(df['Model'])))
        df = df.reset_index()
        idx = df.groupby('tar_id')['we'].idxmin()
        df = df.loc[idx].reset_index(drop=True)
        df['Model'] = unified_model_name
        self.model_names_list = [unified_model_name]
        return df
        
        
    def process_dataset_df(self):
        df = self.dataset_df
        baseline_model_df = df[df['Model'] == self.cfg.baseline_model_name]
        df = df[df['Model'] != self.cfg.baseline_model_name]
        tar_id_to_baseline_we = baseline_model_df.set_index('tar_id')['we'].to_dict()
        df['baseline_we'] = df['tar_id'].map(tar_id_to_baseline_we)
        
        if self.cfg.get('unify_models', False):
            df = self.unify_models(df)
        
        if self.cfg.labeling == 'HardBinaryRouterLabels':
            df['RouterLabel'] = (df['we'] < df['baseline_we']).astype(int)  # TODO: add we-difference buffer for less noisy labels
            
            dataset_columns = ['tar_id', 'features_path', 'w', 'Model', 'we', 'instruction_type', 'RouterLabel', 'baseline_we']
            df = df[dataset_columns]

            df = df[((df['RouterLabel'] == 1) &
                     ((df['we'] / df['w']) < self.cfg.max_sample_wer)) |
                    (df['RouterLabel'] == 0)]
             
            # tar_id_has_good_model = df.groupby('tar_id')['RouterLabel'].max() == 1
            # no_good_model_tar_ids = tar_id_has_good_model[~tar_id_has_good_model].index.to_list()
            # df = df[~df['tar_id'].isin(no_good_model_tar_ids)]
            
        elif self.cfg.labeling == 'HardBinaryRouterLabelsWithBuffer':

            df['wer'] = df['we'] / df['w']
            df['baseline_wer'] = df['baseline_we'] / df['w']
            df = df[((df['wer'] < self.cfg.good_wer_threshold) & 
                    (df['baseline_wer'] > self.cfg.bad_wer_threshold) 
                    | 
                    (df['baseline_wer'] < self.cfg.good_wer_threshold) &
                    (df['wer'] > self.cfg.bad_wer_threshold))]
                
            df['RouterLabel'] = (df['we'] < df['baseline_we']).astype(int)  # TODO: add we-difference buffer for less noisy labels
            
            dataset_columns = ['tar_id', 'features_path', 'w', 'Model', 'we', 'instruction_type', 'RouterLabel', 'baseline_we']
            df = df[dataset_columns]
            
        else:
            raise NotImplemented(f'Currently only supporting "HardBinaryRouterLabels"')
        
        print('===============================')
        print(df['RouterLabel'].value_counts())
        print('===============================')
        
        self.dataset_df = df        

    def __len__(self):
        len_ = self.cfg.get('len_', None)
        if len_ is not None:
            return len_
        
        num_samples = len(self.dataset_df)
        if self.cfg.get('use_mini_batch_per_model', False):
            num_models_per_batch = self.cfg.num_models_per_batch            
            per_model_mini_batch_size = self.cfg.per_model_mini_batch_size
            full_batch_size = num_models_per_batch * per_model_mini_batch_size 
            return num_samples // full_batch_size
        else:
            return num_samples
    
    @staticmethod
    def get_sample_from_row(row):
        tar_id = row['tar_id']
        x = torch.tensor(np.load(row['features_path']), dtype=torch.float32)
        x = x.T  # TODO: This needs to be configurable!!!
        # x = torch.rand(512, random.randint(100, 499), dtype=torch.float32)
        router_label = torch.tensor(row['RouterLabel']).to(torch.int32)
        model_name = row['Model']
        instruction_type = row['instruction_type']
        w = torch.tensor(row['w']).to(torch.int32)
        we = torch.tensor(row['we']).to(torch.int32)
        baseline_we = torch.tensor(row['baseline_we']).to(torch.float32)
        return tar_id, x, router_label, model_name, instruction_type, w, we, baseline_we


    def __getitem__(self, idx):
        
        if self.cfg.get('use_mini_batch_per_model', False):
            num_models_per_batch = self.cfg.num_models_per_batch
            models_for_batch = random.sample(self.model_names_list, num_models_per_batch)

            per_model_mini_batch_size = self.cfg.per_model_mini_batch_size
            full_batch_df = self.dataset_df[self.dataset_df['Model'].isin(models_for_batch)].groupby('Model').sample(n=per_model_mini_batch_size)
            
            full_batch_size = num_models_per_batch * per_model_mini_batch_size
            assert len(full_batch_df) == full_batch_size, f'{len(full_batch_df)}, {full_batch_size}'
            batch_samples = [self.get_sample_from_row(full_batch_df.iloc[i]) for i in range(full_batch_size)]
            batch = self.collate_fn(batch_samples)
            return batch

        else:
            row = self.dataset_df.iloc[idx]
            return self.get_sample_from_row(row)

    @staticmethod
    def already_batched_collate_fn(batch):
        return batch[0]
    
    @staticmethod
    def collate_fn(batch):
        tar_id, x, router_label, model_name, instruction_type, w, we, baseline_we = zip(*batch)

        dim_0 = x[0].shape[0]
        max_size = max(x_.shape[1] for x_ in x)
        x = torch.stack([torch.cat([x_, torch.zeros((dim_0, max_size - x_.shape[1]))], dim=1) for x_ in x], dim=0)
        router_label = torch.stack(router_label)
        w = torch.stack(w)
        we = torch.stack(we)
        baseline_we = torch.stack(baseline_we)
        return tar_id, x, router_label, model_name, instruction_type, w, we, baseline_we

