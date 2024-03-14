import os
import shutil
from collections import defaultdict

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from AI_template.AI_common.models import beat_segmentation
from AI_common.measure import accuracy, miou, fbeta, MarginalMetric
from AI_template.AI_common.operation import print_confusion_matrix
from AI_template.AI_common.trainer import Trainer as ai_Trainer

from src.loss import get_loss
from src.common.setup import Setter, Configure
from src.common.path import check_dir
from src.fileio import DatasetParser_v2
from src.data import ECG_dataset, combine_label, Collate_seq
from src.utils.opt import CosineAnnealingWarmUpRestarts
from src.model.network import BeatSegment
from src.postprocess import fetch_class_multi_to_binary, smoothing, accumulate_morphed_10sec, Dx_naming
from src.verify import plot_compare, plot_seg, set_seeds


def run(data_directory, run_directory, mode='test'):
    print('\nRun for stage 1...')
    setter = Setter.get()

    cp_direc = os.path.join(run_directory, '_checkpoints')
    output_direc = os.path.join(run_directory, '_outputs')
    if not os.path.isdir(cp_direc):             check_dir(cp_direc)
    if not os.path.isdir(output_direc):             check_dir(output_direc)

    model_save_path = os.path.join(cp_direc, 'ECG_seg_checkpoint.pt')
    result_path = os.path.join(output_direc, 'ECG_seg_result.npy')
    output_path = os.path.join(output_direc, 'ECG_seg_output.npy')

    Configure.init_cfg()
    if mode == 'test':
        cfg = Configure.get_cfg(os.path.join(run_directory, '_checkpoints/config_stage_1.json'))
    elif mode == 'train':
        cfg = Configure.get_cfg('./_config/config_stage_1.json')
    elif mode == 'inference':
        cfg = Configure.get_cfg(os.path.join(run_directory, '_checkpoints/config_stage_1.json'))

    # --------------------------------- Load data --------------------------------- #
    parser = DatasetParser_v2(data_directory, seed=setter.SEED)
    # ds = parser.parse_from_path(path=data_directory)
    if mode == 'train':
        train_dataset = ECG_dataset(**parser.parse(mode='train', long=cfg.run.pretrain), mode='train')
        valid_dataset = ECG_dataset(**parser.parse(mode='valid', long=cfg.run.pretrain), mode='valid')
        test_dataset = ECG_dataset(**parser.parse(mode='test', long=cfg.run.pretrain), mode='test')
        loader_dict = {'train': DataLoader(train_dataset, batch_size=cfg.run.batch_size_train, shuffle=True,
                                           collate_fn=Collate_seq(sequence=['noise_aug', 'p_tachy']).wrap()),
                       'valid': DataLoader(valid_dataset, batch_size=cfg.run.batch_size_eval, shuffle=False,
                                           collate_fn=Collate_seq(()).wrap()),
                       'test': DataLoader(test_dataset, batch_size=cfg.run.batch_size_eval, shuffle=False,
                                          collate_fn=Collate_seq(()).wrap())}
    elif mode == 'test':
        test_dataset = ECG_dataset(**parser.parse(mode='test'), mode='test')
        loader_dict = {'test': DataLoader(test_dataset, batch_size=cfg.run.batch_size_eval, shuffle=False,
                                          collate_fn=Collate_seq(()).wrap())}
    elif mode == 'inference':
        # Not implemented
        inference_dataset = ECG_dataset(**parser.parse(mode='test'), mode='test')
        loader_dict = {'inference': DataLoader(inference_dataset, batch_size=cfg.run.batch_size_eval, shuffle=False,
                                          collate_fn=Collate_seq(()).wrap())}

    result, out_put = core(mode, loader_dict=loader_dict, model_save_path=model_save_path, pretrained=cfg.run.pretrain)
    if result is None:
        return
    if mode == 'inference':
        np.save(result_path, result)
        np.save(output_path, out_put)
    else:
        result_mat = np.zeros((cfg.model.ch_out, cfg.model.ch_out))
        for o, l in zip(result, test_dataset.y.detach().numpy()):
            result_mat += MarginalMetric(prediction=o, label=l, num_class=cfg.model.ch_out).marginal_mat

        print_confusion_matrix(result_mat, class_name=['B', 'P', 'T', 'N', 'A', 'V', 'Q'])
        print(f'\n\tBeat Accuracy: {100 * result_mat[1:, 1:].sum() / result_mat.sum():.2f} %')
        print(f'\tF1-score: {100 * fbeta(result_mat[1:, 1:], beta=1, num_classes=cfg.model.ch_out-1):.2f} %')

        np.save(result_path, result)
        np.save(output_path, out_put)
    return result, out_put      # nd-array, [B, dim]


def core(mode, loader_dict, model_save_path, pretrained=False):
    cfg = Configure.get_cfg('')
    setter = Setter.get()
    loader_dict = defaultdict(**loader_dict)

    if mode == 'train':
        assert mode in loader_dict.keys(), 'Train loader must be provided on "train" mode. Valid loader is optional.'
        train_loader, valid_loader, test_loader = loader_dict['train'], loader_dict['valid'], loader_dict['test']
    elif mode == 'test':
        test_loader = loader_dict['test']
    else:
        inference_loader = loader_dict['inference']

    # --------------------------------- Modeling --------------------------------- #
    model = BeatSegment(ch_in=cfg.model.ch_in, ch_out=cfg.model.ch_out, width=cfg.model.width,
                        kernel_size=cfg.model.kernel_size, depth=cfg.model.depth, order=cfg.model.order,
                        stride=cfg.model.stride, decoding=cfg.model.decoding,
                        se_bias=cfg.model.se_bias, expanding=cfg.model.expanding)

    tot_epoch = cfg.run.num_epochs
    opt = torch.optim.Adam(model.parameters(), lr=cfg.run.learning_rate, weight_decay=cfg.run.weight_decay)
    scheduler = CosineAnnealingWarmUpRestarts(opt, eta_max=cfg.run.learning_rate, gamma=0.5,
                                              T_up=int(0.1 * tot_epoch), T_0=tot_epoch)

    if pretrained:
        model.load_state_dict(beat_segmentation(True).state_dict())
        old_conv = model.enc[0][0][0]
        old_w = old_conv.weight

        ch_tgt, ch_old, ker = old_w.shape
        ch_new = cfg.data.input_channels
        ch_diff = ch_new - ch_old
        w_pad = [torch.zeros_like(old_w)] * (ch_diff // 2)

        old_conv.in_channels = ch_new
        old_conv.weight.data = torch.cat((w_pad + [old_w] + w_pad), dim=1)
        model.enc[0][0][0] = old_conv

        f_fine = 2
        tot_epoch //= f_fine
        opt = torch.optim.Adam(model.parameters(), lr=cfg.run.learning_rate/f_fine, weight_decay=cfg.run.weight_decay)
        scheduler = CosineAnnealingWarmUpRestarts(opt, eta_max=cfg.run.learning_rate/f_fine, gamma=0.5,
                                                  T_up=int(0.1 * tot_epoch), T_0=tot_epoch)

    trainer = Trainer_stg1(model=model, device=setter.device, mode=mode, optimizer=opt, scheduler=scheduler)

    # --------------------------------- Training & Validation --------------------------------- #
    # train
    if mode == 'train':
        print('\nTrain model')
        trainer.run(n_epoch=tot_epoch, loader=train_loader, valid_loader=valid_loader)
        torch.save(trainer.weight_buffer['weight'][-1], model_save_path)
        shutil.copy('./_config/config_stage_1.json',
                    os.path.join(os.path.split(model_save_path)[0], 'config_stage_1.json'))

    if mode != 'inference':
        if test_loader is None: return

    print('\nTest model')
    if model_save_path == 'server':
        trainer.model = beat_segmentation(pretrained=True)
    else:
        trainer.apply_weights(filename=model_save_path if mode != 'train' else None)

    if mode == 'inference':
        result = np.concatenate(trainer.inference(inference_loader)).argmax(1)
        out_put = np.concatenate(trainer.inference(inference_loader))

    else:
        result = np.concatenate(trainer.inference(test_loader)).argmax(1)
        out_put = np.concatenate(trainer.inference(test_loader))

    return result, out_put


def export_archi(run_direc):
    setter = Setter.get()
    cfg_path = os.path.join(run_direc, '_checkpoints/config_stage_1.json')
    cfg = Configure.get_cfg(cfg_path)

    model = BeatSegment(ch_in=cfg.model.ch_in, ch_out=cfg.model.ch_out, width=cfg.model.width,
                        kernel_size=cfg.model.kernel_size, depth=cfg.model.depth, order=cfg.model.order,
                        stride=cfg.model.stride, decoding=cfg.model.decoding,
                        se_bias=cfg.model.se_bias, expanding=False)
    trainer = Trainer_stg1(model=model, device=setter.device, mode='test')
    trainer.apply_weights(filename=cfg_path.replace('config_stage_1.json', 'ECG_seg_checkpoint.pt'))

    onnx_path = cfg_path.replace('config_stage_1.json', 'ECG_seg_arch.onnx')
    torch.onnx.export(model, torch.empty(1, 1, 2560, dtype=torch.float32).to(setter.device),
                      onnx_path, opset_version=11)


def pre_annotation(data_directory, run_directory):
    setter = Setter.get()
    loader = None       # custom
    model = AI_common.models.benchmark.beat_segmentation(pretrained=True)

    trainer = Trainer_stg1(model, setter.device, mode='test', optimizer=None, scheduler=None)
    result = trainer.inference(loader)

    seg_to_anno = [Dx_naming(res) for res in result]

    # Save as txt format
    if 'apc' in data_directory:
        cls_name = 'apc'
    elif 'vpc' in data_directory:
        cls_name = 'vpc'
    elif 'nsr' in data_directory:
        cls_name = 'nsr'
    else:
        cls_name = 'UNKNOWN'

    save_path = os.path.join(run_directory, cls_name)
    if not os.path.isdir(save_path):            os.mkdir(save_path)

    for i in range(len(fn)):
        with open(os.path.join(save_path, fn[i] + '-annotations.txt'), 'w+', encoding='utf-8') as lf:
            lf.write('\n'.join(seg_to_anno[i][0]))


# --------------------------------- Trainer --------------------------------- #
class Trainer_stg1(ai_Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(Trainer_stg1, self).__init__(model, device, mode, optimizer, scheduler)

    def train_step(self, batch: dict):
        x, y, fn = batch["x"], batch['y'].squeeze(1), batch['fn']
        # print('train_step')
        # print(x)
        # print(y)
        # print(fn)

        out = self.model(x)
        # print(out)
        if isinstance(out, tuple): out = out[-1]
        out_proba = F.softmax(out, 1)
        loss = get_loss(out, y, run_cfg=None) + 1e-3 * get_loss(out_proba, y, run_cfg=None, mode='topo')
        out_proba_, y_ = self._to_cpu(out_proba), self._to_cpu(y)
        m_iou = miou(outputs=out_proba_.argmax(1).numpy(), labels=y_.numpy()).mean()
        acc = accuracy(predict=out_proba_.argmax(1).numpy(), true=y_.numpy()).mean()
        return {'loss': loss, 'output': out, 'miou': m_iou, 'acc': acc}

    def test_step(self, batch: dict):
        x, fn = batch["x"], batch['fn']
        y = None if batch['y'] is None else batch['y'].squeeze(1)
        out = self.model(x)
        if isinstance(out, tuple): out = out[-1]
        if y is None:
            return {'loss': 0, 'output': out, 'miou': 0, 'acc': 0}
        out_proba = F.softmax(out, 1)
        loss = get_loss(out, y, run_cfg=None)
        out_proba_, y_ = self._to_cpu(out_proba), self._to_cpu(y)
        m_iou = miou(outputs=out_proba_.argmax(1).numpy(), labels=y_.numpy()).mean()
        acc = accuracy(predict=out_proba_.argmax(1).numpy(), true=y_.numpy()).mean()
        return {'loss': loss, 'output': out, 'miou': m_iou, 'acc': acc}


# --------------------------------- Scheduler --------------------------------- #
def post(cfg, output):
    # Reshape to remove dummy dimension
    test_results = output.reshape(-1, 2500)
    test_results = test_results.astype(np.float32)

    # Split and duplicate data for each class (w/o baseline)
    # Output: tuple(arrays of NSR, VPC, APC) - float
    bin_data = fetch_class_multi_to_binary(data=test_results, num_label=cfg.model.ch_out, baseline_idx=0)

    # Morphological smoothing. 10 sec 단위로 동작함.
    morphed_data = smoothing(bin_data, operation=cfg.post.operation, kernel_size=cfg.post.kernel_size)
    morphed = accumulate_morphed_10sec(morphed_data)

    cleared = cleaning(label=morphed)

    return morphed.astype(np.int32), cleared.squeeze().astype(np.int32)
