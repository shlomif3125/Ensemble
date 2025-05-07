import random
import torch
from torch.utils.data import Dataset


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
        we = torch.tensor(row['we']).to(torch.float32)

        return tar_id, x, router_label, model_name, we

    @staticmethod
    def collate_fn(batch):
        tar_id, x, router_label, model_name, we = zip(*batch)

        dim_0 = x[0].shape[0]
        max_size = max(x_.shape[1] for x_ in x)
        x = torch.stack([torch.cat([x_, torch.zeros((dim_0, max_size - x_.shape[1]))], dim=1) for x_ in x], dim=0)
        router_label = torch.stack(router_label)
        we = torch.stack(we)
        return tar_id, x, router_label, model_name, we


if __name__ == '__main__':
    import pandas as pd
    dataset_df = pd.read_pickle('router_dataset_v1.pkl')
    ds = MultiRouterDatasetV0(dataset_df)
    tar_id, x, router_label, model_name, we = ds[0]


        