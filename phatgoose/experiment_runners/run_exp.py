import sys
sys.path.append('/home/shlomi.fenster/notebooks/nemo_stuff/Onboarding/Ensemble/phatgoose')
sys.path.append('/home/shlomi.fenster/MyNeMo/q-NeMo')
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from data_utils.data import MultiRouterDataset
from model_utils.models import MultiRouterModel
from nemo.core.config import hydra_runner


@hydra_runner(config_path="../configs", config_name="config")
def main(cfg):
    
    data_cfg = cfg.data
    train_ds = MultiRouterDataset(data_cfg.train)
    if data_cfg.train.get('use_mini_batch_per_model', False):
        collate_fn = train_ds.already_batched_collate_fn
        batch_size = 1
    else:
        collate_fn = train_ds.collate_fn
        batch_size = data_cfg.train.batch_size
    
    train_dl = DataLoader(train_ds, batch_size, collate_fn=collate_fn, shuffle=True, drop_last=True, 
                          num_workers=data_cfg.num_workers, pin_memory=data_cfg.pin_memory)

    val_dls = []
    for val_cfg in data_cfg.validation.val_sets:
        val_ds = MultiRouterDataset(val_cfg)
        val_dl = DataLoader(val_ds, val_cfg.batch_size, collate_fn=train_ds.collate_fn, shuffle=False, drop_last=False, 
                              num_workers=data_cfg.num_workers, pin_memory=data_cfg.pin_memory)
        val_dls.append(val_dl)


    multi_router_model = MultiRouterModel(cfg.model)

    trainer_cfg = cfg.trainer
    logger = pl.loggers.TensorBoardLogger(save_dir=trainer_cfg.exp_dir)
    trainer = pl.Trainer(max_epochs=trainer_cfg.max_epochs, max_steps=trainer_cfg.max_steps, 
                         val_check_interval=trainer_cfg.val_check_interval, 
                         log_every_n_steps=trainer_cfg.log_every_n_steps,
                         logger=logger)

    trainer.fit(multi_router_model, train_dl, val_dls)
    
if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter
