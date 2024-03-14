# %%
import os, time, csv
import AI_template.AI_common.models
from AI_template.AI_common.trainer import Trainer as ai_Trainer
import numpy as np
import shutil
import torch
from torch.utils.data import DataLoader
from src.common.setup import Setter
from src.common.path import check_dir
from src.fileio import DatasetParser_v2
from src.common.setup import Configure
from src.data import ECG_dataset, Collate_seq
from src.loss import get_mse_loss
from src.utils.opt import CosineAnnealingWarmUpRestarts
import matplotlib.pyplot as plt


def run(data_directory, run_directory, mode='test'):
    print('\nRun for stage 2...')
    setter = Setter.get()

    cp_direc = os.path.join(run_directory, '_checkpoints')
    output_direc = os.path.join(run_directory, '_outputs')
    if not os.path.isdir(cp_direc):             check_dir(cp_direc)
    if not os.path.isdir(output_direc):             check_dir(output_direc)

    model_save_path = os.path.join(cp_direc, 'ECG_reg_checkpoint.pt')
    output_path = os.path.join(output_direc, 'ECG_reg_output.npy')

    Configure.init_cfg()
    if mode == 'train':
        cfg = Configure.get_cfg('./_config/config_stage_2.json')
    else:
        cfg = Configure.get_cfg(os.path.join(run_directory, '_checkpoints/config_stage_2.json'))

    # --------------------------------- Load data --------------------------------- #
    parser = DatasetParser_v2(data_directory, seed=setter.SEED)
    if mode == 'train':
        train_dataset = ECG_dataset(**parser.parse(mode='train'), mode='train')
        valid_dataset = ECG_dataset(**parser.parse(mode='valid'), mode='valid')
        test_dataset = ECG_dataset(**parser.parse(mode='test'), mode='test')
        loader_dict = {'train': DataLoader(train_dataset, batch_size=cfg.run.batch_size_train, shuffle=True,
                                           collate_fn=Collate_seq(sequence=['r_reg'], resamp=cfg.model.dim_in).wrap()),
                       'valid': DataLoader(valid_dataset, batch_size=cfg.run.batch_size_eval, shuffle=False,
                                           collate_fn=Collate_seq(sequence=['r_reg'], resamp=cfg.model.dim_in).wrap()),
                       'test': DataLoader(test_dataset, batch_size=cfg.run.batch_size_eval, shuffle=False,
                                          collate_fn=Collate_seq(sequence=['r_reg'], resamp=cfg.model.dim_in).wrap())}
    elif mode == 'test':
        test_dataset = ECG_dataset(**parser.parse(mode='test'), mode='test')
        loader_dict = {'test': DataLoader(test_dataset, batch_size=cfg.run.batch_size_eval, shuffle=False,
                                          collate_fn=Collate_seq(sequence=['r_reg'], resamp=cfg.model.dim_in).wrap())}
    else:
        # Not implemented
        test_dataset = ECG_dataset(**parser.parse(mode='test'), mode='test')
        loader_dict = {'test': DataLoader(test_dataset, batch_size=cfg.run.batch_size_eval, shuffle=False,
                                          collate_fn=Collate_seq(sequence=['r_reg'], resamp=cfg.model.dim_in).wrap())}
    outputs = core(mode, loader_dict=loader_dict, model_save_path=model_save_path)

    np.save(output_path, outputs)
    return outputs


def core(mode, loader_dict, model_save_path):
    cfg = Configure.get_cfg('')
    setter = Setter.get()
    train_loader, valid_loader, test_loader = None, None, None

    if mode == 'train':
        assert mode in loader_dict.keys(), 'Train loader must be provided on "train" mode. Valid loader is optional.'
        train_loader = loader_dict['train']
        if 'valid' in loader_dict.keys(): valid_loader = loader_dict['valid']

    elif mode in ['test', 'inference']:
        assert 'test' in loader_dict.keys(), 'Test loader must be provided on "test" mode.'
        test_loader = loader_dict['test']

    # --------------------------------- Modeling --------------------------------- #
    model = AI_common.models.RPeakRegress(dim_in=cfg.model.dim_in, ch_in=cfg.model.ch_in, width=cfg.model.width,
                                          kernel_size=cfg.model.kernel_size, depth=cfg.model.depth,
                                          stride=cfg.model.stride, order=cfg.model.order,
                                          head_depth=cfg.model.head_depth, embedding_dims=cfg.model.embedding_dims,
                                          se_bias=cfg.model.se_bias)

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=cfg.run.learning_rate, weight_decay=cfg.run.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(optimizer, eta_max=cfg.run.learning_rate, gamma=0.5,
                                              T_up=int(0.1 * cfg.run.num_epochs), T_0=cfg.run.num_epochs)

    trainer = Trainer_stg2(model, setter.device, mode=mode, optimizer=optimizer, scheduler=scheduler)

    # --------------------------------- Training & Validation --------------------------------- #
    # train
    if mode == 'train':
        print('\nTrain model')
        trainer.run(n_epoch=cfg.run.num_epochs, loader=train_loader, valid_loader=valid_loader)
        torch.save(trainer.weight_buffer['weight'][-1], model_save_path)
        shutil.copy('./_config/config_stage_1.json',
                    os.path.join(os.path.split(model_save_path)[0], 'config_stage_1.json'))

    print('\nTest model')
    if model_save_path == 'server':
        trainer.model = AI_common.models.rpeak_regression(pretrained=True)
    else:
        trainer.apply_weights(filename=model_save_path if mode != 'train' else None)
    return np.concatenate(trainer.inference(test_loader)).squeeze()


class Trainer_stg2(ai_Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(Trainer_stg2, self).__init__(model, device, mode, optimizer, scheduler)

    def train_step(self, batch: dict):
        w, r_loc = batch['w'], batch['r_loc']
        pred = self.model(w)
        loss = get_mse_loss(pred, r_loc)
        return {'loss': loss, 'output': pred}

    def test_step(self, batch: dict):
        w = batch['w']
        r_loc = None if batch['r_loc'] is None else batch['r_loc'].squeeze()
        pred = self.model(w)
        if r_loc is None:
            onoff = None if batch['onoff'] is None else batch['onoff']
            return {'loss': 0, 'output': self.model.clipper(pred), 'output_onoff': onoff}
        loss = get_mse_loss(pred, r_loc)
        return {'loss': loss, 'output': self.model.clipper(pred)}
