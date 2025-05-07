import random
import pandas as pd
import torch
from torch.utils.data import Dataset


class MultiRouterDataset(Dataset):
    def __init__(self, data_cfg):
        self.cfg = data_cfg
        self.dataset_df = self.create_dataset_df()
        self.model_names_list = sorted(set(self.dataset_df['Model']))
        
    def create_dataset_df(self):
        df = pd.read_pickle(self.cfg.manifest_filepath)
        baseline_model_df = df[df['Model'] == 'baseline_model_200K']
        df = df[df['Model'] != 'baseline_model_200K']
        tar_id_to_baseline_we = baseline_model_df.set_index('tar_id')['we'].to_dict()
        df['baseline_we'] = df['tar_id'].map(tar_id_to_baseline_we)
        if self.cfg.labeling == 'HardBinaryRouterLabels':
            df['RouterLabel'] = (df['we'] < df['baseline_we']).astype(int)
            
            dataset_columns = ['tar_id', 'w', 'Model', 'we', 'instruction_type', 'RouterLabel']
            df = df[dataset_columns]

            df = df[((df['RouterLabel'] == 1) &
                     ((df['we'] / df['w']) < self.cfg.max_sample_wer)) |
                    (df['RouterLabel'] == 0)]

            tar_id_has_good_model = df.groupby('tar_id')['RouterLabel'].max() == 1
            no_good_model_tar_ids = tar_id_has_good_model[~tar_id_has_good_model].index.to_list()
            df = df[~df['tar_id'].isin(no_good_model_tar_ids)]
        else:
            raise NotImplemented(f'Currently only supporting "HardBinaryRouterLabels"')
        
        return df        

    def __len__(self):
        return len(self.dataset_df)
    
    @staticmethod
    def get_sample_from_row(row):
        tar_id = row['tar_id']
        # x = torch.tensor(np.load(row['input_tensor_path']), dtype=torch.float32)
        x = torch.rand(512, random.randint(100, 499), dtype=torch.float32)
        router_label = torch.tensor(row['RouterLabel']).to(torch.int32)
        model_name = row['Model']
        instruction_type = row['instruction_type']
        w = torch.tensor(row['w']).to(torch.int32)
        we = torch.tensor(row['we']).to(torch.int32)
        return tar_id, x, router_label, model_name, instruction_type, w, we


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
        tar_id, x, router_label, model_name, instruction_type, w, we = zip(*batch)

        dim_0 = x[0].shape[0]
        max_size = max(x_.shape[1] for x_ in x)
        x = torch.stack([torch.cat([x_, torch.zeros((dim_0, max_size - x_.shape[1]))], dim=1) for x_ in x], dim=0)
        router_label = torch.stack(router_label)
        w = torch.stack(w)
        we = torch.stack(we)
        return tar_id, x, router_label, model_name, instruction_type, w, we


class MultiRouterDatasetV0(Dataset):
    def __init__(self, dataset_df):
        self.dataset_df = dataset_df

    def __len__(self):
        return len(self.dataset_df)

    def __getitem__(self, idx):
        row = self.dataset_df.iloc[idx]
        tar_id = row['tar_id']
        # x = torch.tensor(np.load(row['input_tensor_path']), dtype=torch.float32)
        x = torch.rand(512, random.randint(100, 499), dtype=torch.float32)
        router_label = torch.tensor(row['RouterLabel']).to(torch.int32)
        model_name = row['Model']
        instruction_type = row['instruction_type']
        w = torch.tensor(row['w']).to(torch.int32)
        we = torch.tensor(row['we']).to(torch.int32)

        return tar_id, x, router_label, model_name, instruction_type, w, we

    @staticmethod
    def collate_fn(batch):
        tar_id, x, router_label, model_name, instruction_type, w, we = zip(*batch)

        dim_0 = x[0].shape[0]
        max_size = max(x_.shape[1] for x_ in x)
        x = torch.stack([torch.cat([x_, torch.zeros((dim_0, max_size - x_.shape[1]))], dim=1) for x_ in x], dim=0)
        router_label = torch.stack(router_label)
        w = torch.stack(w)
        we = torch.stack(we)
        return tar_id, x, router_label, model_name, instruction_type, w, we


if __name__ == '__main__':
    import pandas as pd
    dataset_df = pd.read_pickle('router_dataset_v1.pkl')
    ds = MultiRouterDatasetV0(dataset_df)
    tar_id, x, router_label, model_name, instruction_type, w, we = ds[0]


        