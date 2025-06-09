import sys
sys.path.append('/home/shlomi.fenster/notebooks/nemo_stuff/Onboarding/Ensemble/phatgoose')
sys.path.append('/home/shlomi.fenster/MyNeMo/q-NeMo')
import os
import pandas as pd
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from data_utils.data import MultiRouterDataset
from model_utils.models import MultiRouterModel
from nemo.core.config import hydra_runner
from omegaconf import OmegaConf, DictConfig, open_dict
import logging

OmegaConf.register_new_resolver("eval", eval)

def list_of_strings_to_string(str_lst):
    return '__'.join(str_lst)

OmegaConf.register_new_resolver("list_of_strings_to_string", list_of_strings_to_string)


@hydra_runner(config_path="../configs", config_name="config")
def main(cfg):
    
    logging.info('Started main...')
    
    logging.info('Creating train dataset...')
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
    val_dl_names = []
    val_dl_types = []
 
    logging.info('Creating val dataset...')   
    val_cfg = data_cfg.validation
    val_ds = MultiRouterDataset(val_cfg)
    val_dl = DataLoader(val_ds, val_cfg.batch_size, collate_fn=train_ds.collate_fn, shuffle=False, drop_last=False, 
                            num_workers=data_cfg.num_workers, pin_memory=data_cfg.pin_memory)
    val_dls.append(val_dl)
    val_dl_names.append('val')
    val_dl_types.append('val')
    
    # for test_cfg in data_cfg.test.test_sets:
    #     test_ds = MultiRouterDataset(test_cfg)
    #     test_dl = DataLoader(test_ds, val_cfg.batch_size, collate_fn=train_ds.collate_fn, shuffle=False, drop_last=False, 
    #                          num_workers=data_cfg.num_workers, pin_memory=data_cfg.pin_memory)
    #     val_dls.append(test_dl)
    #     val_dl_names.append(test_cfg.name)
    #     val_dl_types.append('test')
    
    logging.info('Creating test dataset...')
    test_cfg = data_cfg.test
    test_ds = MultiRouterDataset(test_cfg)
    test_dl = DataLoader(test_ds, val_cfg.batch_size, collate_fn=train_ds.collate_fn, shuffle=False, drop_last=False, 
                            num_workers=data_cfg.num_workers, pin_memory=data_cfg.pin_memory)
    val_dls.append(test_dl)
    val_dl_names.append('test')
    val_dl_types.append('test')

    model_cfg = cfg.model
    with open_dict(model_cfg):
        model_cfg.val_dl_names = val_dl_names
        model_cfg.val_dl_types = val_dl_types
        
        model_names = sorted(set(train_ds.dataset_df['Model']))     
        model_cfg.model_names_list = model_names            

    logging.info('Creating model...')
    multi_router_model = MultiRouterModel(cfg.model)

    trainer_cfg = cfg.trainer
    logger = pl.loggers.TensorBoardLogger(save_dir=trainer_cfg.exp_dir)
    callbacks = [ModelCheckpoint(**trainer_cfg.checkpoint_callback_params),
                 LearningRateMonitor()]
    
    logging.info('Creating trainer...')
    trainer = pl.Trainer(max_epochs=trainer_cfg.max_epochs, max_steps=trainer_cfg.max_steps, 
                         val_check_interval=trainer_cfg.val_check_interval, 
                         log_every_n_steps=trainer_cfg.log_every_n_steps,
                         logger=logger, callbacks=callbacks)
    
    save_cfg_path = os.path.join(trainer_cfg.exp_dir, 'cfg.yaml')
    logging.info(f'Saving cfg to yaml in {save_cfg_path}')
    os.makedirs(os.path.dirname(save_cfg_path), exist_ok=True)
    OmegaConf.save(cfg, save_cfg_path, resolve=True)

    trainer.fit(multi_router_model, train_dl, val_dls)
    

if __name__ == '__main__':
    main()  # noqa pylint: disable=no-value-for-parameter

