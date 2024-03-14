from functools import wraps
from collections import defaultdict
from typing import Union, Tuple, List, Callable
from abc import ABC, abstractmethod
from collections import defaultdict
from functools import wraps
from scipy import signal
import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from einops import repeat, reduce
from src.augmentation import noise_augmentation, pseudo_tachy
from AI_template.AI_common import models, trainer, data
from AI_template.AI_common.operation import label_to_onoff, sanity_check, onoff_to_label


def core(x: Union[np.ndarray, torch.Tensor, list], pretrain):

    assert len(x.shape) == 1, f'Expect 1-D array, but receive {len(x.shape)}-D array.'

    device = torch.device(f'cuda:0' if torch.cuda.is_available() else 'cpu')
    x = x.copy()

    inference_dataset = ECG_dataset(x=x.reshape(1, -1))
    inference_loader = DataLoader(inference_dataset, batch_size=1, shuffle=False,
                                  collate_fn=BeatCollateSeq(sequence=['split']).wrap())

    # --------------------------------- Inference & Postprocess @ stage 1 --------------------------------- #
    model = models.beat_segmentation(pretrain)
    trainer1 = Trainer_stg1(model=model, device=device, mode='test')

    out_put = np.concatenate(trainer1.inference(inference_loader))
    prediction_stg1 = np.concatenate(trainer1.inference(inference_loader)).argmax(1)

    # postprocessing
    cursor, fin_dict = 0, defaultdict(list)
    for col in inference_loader.__iter__():
        n = col['x'].__len__()
        fin_dict['x'].append(col['raw'])
        pred_onoff = label_to_onoff(prediction_stg1[cursor:cursor + n], sense=5)
        # pred_onoff = [sanity_check(oo) for oo in pred_onoff]
        pred_onoff = recursive_peak_merge(pred_onoff, len(pred_onoff) // 2, 250, 2)[0]
        try:
            if not pred_onoff:
                raise np.AxisError(0)
            fin_dict['onoff'].append(pred_onoff[pred_onoff.max(1) < len(col['raw'])])
        except np.AxisError:
            fin_dict['onoff'].append(None)
        except ValueError:
            fin_dict['onoff'].append(pred_onoff[pred_onoff.max(1) < len(col['raw'])])
        fin_dict['y'].append(onoff_to_label(np.array(pred_onoff), length=len(col['raw'])).tolist())
        fin_dict['fn'].append(col['fn'])
        cursor += n

    try:
        if set(fin_dict['onoff']) == {None}: return []
    except TypeError:
        pass

    # --------------------------------- Load data @ stage 2 --------------------------------- #
    inference_dataset2 = ECG_dataset(x=np.stack(fin_dict['x']), y=np.stack(fin_dict['y']),
                                     fn=fin_dict['fn'], onoff=fin_dict['onoff'])
    inference_loader2 = DataLoader(inference_dataset2, batch_size=1, shuffle=False,
                                   collate_fn=BeatCollateSeq(sequence=['r_reg'], zit=False, resamp=64).wrap())

    # --------------------------------- Inference & Postprocess @ stage 2 --------------------------------- #
    model = models.rpeak_regression(True)
    trainer2 = Trainer_stg2(model=model, device=device, mode='test')
    prediction_stg2 = np.concatenate(trainer2.inference(inference_loader2)).squeeze()

    # postprocessing
    cursor = 0
    for i, onoff in enumerate(inference_dataset2.onoff):
        n = onoff.__len__()
        r_loc = prediction_stg2[cursor:cursor + n]
        r_idx = onoff[:, 0] + (onoff[:, 1] - onoff[:, 0]) * r_loc
        fin_onoff = np.concatenate((onoff, r_idx.reshape(-1, 1)), axis=1).astype(int)
        fin_dict['fin_onoff'].append(fin_onoff[fin_onoff[:, -1] < len(fin_dict['x'][i])])
        cursor += n

    # confidence lv 측정을 위해 output export
    if fin_dict['fin_onoff']:
        return [onoff.tolist() for onoff in fin_dict['fin_onoff']][0], out_put
    else:
        return []


def recursive_peak_merge(qrs_indexes, half_index, sampling_rate, overlapped_sec):
    """ merge overlapped 10sec data.
    Args:
         qrs_indexes: qrs info, N x B x (on off rp cls). N is # of 10 sec data, B is # of beats in a data.
         half_index: half # of 10 sec data.
         sampling_rate: # of pnts in a second.
         overlapped_sec: data is merged according to this param.
    """

    n_overlapped_samples = sampling_rate * overlapped_sec  # 500
    merged_qrs_indexes = np.empty((0, 3), int)  # dummy

    if len(qrs_indexes) == 1:
        return np.array(qrs_indexes)  # data가 1개라 merge 할 게 없음.
    elif len(qrs_indexes) == 2:
        if len(qrs_indexes[0]) > 0:
            merged_qrs_indexes = np.array(qrs_indexes[0])  # 1번째 data에 뭐가 있는 경우
        if len(qrs_indexes[1]) == 0:
            return np.expand_dims(merged_qrs_indexes, axis=0)  # 2번째 data가 빈 경우 땡큐

        shift = int(half_index * (sampling_rate * 10 - n_overlapped_samples))  # half_index는 아마 1?
        shifted_peak_indexes = np.array(qrs_indexes[1])  # 2번째 data
        shifted_peak_indexes[:, 0:2] = shifted_peak_indexes[:, 0:2] + shift

        # 누적된 신호에서 오버랩 후보 영역을 찾는다
        overlapped_pos = np.where(merged_qrs_indexes[:, 1] >= shift - 10)[0]
        overlapped_indexes = merged_qrs_indexes[overlapped_pos]
        # 현재 10초 신호에서 overlap 가능성이 있는 앞부분만 떼어내기
        shifted_overlapped_pos = np.where(np.array(qrs_indexes[1])[:, 0] < n_overlapped_samples)[0]
        shifted_overlapped_indexes = shifted_peak_indexes[shifted_overlapped_pos]

        if len(overlapped_indexes) == 0 or len(shifted_overlapped_indexes) == 0:  # overlap이 없으면 그냥 합치기
            if len(merged_qrs_indexes) > 0 and len(shifted_peak_indexes) > 0:
                merged_qrs_indexes = np.concatenate((merged_qrs_indexes, shifted_peak_indexes), axis=0)
            elif len(shifted_peak_indexes) > 0:
                merged_qrs_indexes = shifted_peak_indexes
        else:  # overlap이 있는 경우
            # 겹치는 qrs 찾기
            # qrs 중심 거리가 기존에 누적된 qrs 중심 거리와 30 이내라면 중복 qrs로 취급
            duplicated = [False] * len(shifted_overlapped_indexes)
            for j, shifted_index in enumerate(shifted_overlapped_indexes):
                shifted_qrs_center = shifted_index[0] + (shifted_index[1] - shifted_index[0]) / 2
                for k, overlapped_index in enumerate(overlapped_indexes):
                    overlapped_qrs_center = overlapped_index[0] + (overlapped_index[1] - overlapped_index[0]) / 2
                    if abs(shifted_qrs_center - overlapped_qrs_center) < 30:
                        duplicated[j] = True
                        break
                    if not (overlapped_index[1] < shifted_index[0] or shifted_index[1] < overlapped_index[0]):
                        duplicated[j] = True
                        break
            # overlap 구간에서 중복되지 않는 모든 qrs 붙이기
            overlapped = np.concatenate(
                (overlapped_indexes, shifted_overlapped_indexes[np.where(np.array(duplicated) == False)]), axis=0)
            overlapped = sorted(overlapped, key=lambda x: x[0])

            merged_qrs_indexes = np.concatenate(
                (merged_qrs_indexes[:overlapped_pos[0]],
                 overlapped,
                 shifted_peak_indexes[shifted_overlapped_pos[-1] + 1:]),
                axis=0)

        return np.expand_dims(merged_qrs_indexes, axis=0)
    else:
        # half... recursive
        n = len(qrs_indexes) // 2
        m1 = recursive_peak_merge(qrs_indexes[:n], n // 2, sampling_rate, overlapped_sec)
        m2 = recursive_peak_merge(qrs_indexes[n:], (len(qrs_indexes) - n) // 2, sampling_rate, overlapped_sec)
        m1 = m1.tolist()
        m2 = m2.tolist()
        m = m1 + m2
        return recursive_peak_merge(m, n, sampling_rate, overlapped_sec)



# --------------------------------- Dataset --------------------------------- #
class ECG_dataset(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.x = torch.tensor(kwargs['x'], dtype=torch.float32)
        self.y = torch.tensor(kwargs['y'], dtype=torch.long) if self.attr('y') is not None else None
        self.fn = kwargs['fn'] if self.attr('fn') is not None else None
        self.rp = kwargs['rp'] if self.attr('rp') is not None else None
        self.onoff = kwargs['onoff'] if self.attr('onoff') is not None else None

        self.mode = kwargs['mode'] if 'mode' in kwargs.keys() else ''
        print(f'\t # of {self.mode} data: %d' % len(self.x))

    def __len__(self):
        return len(self.x)

    def attr(self, var_name):
        if var_name in self.kwargs.keys():
            return self.kwargs[var_name]

    def __getitem__(self, idx):
        batch_dict = defaultdict(None)
        batch_dict['x'] = self.x[idx].unsqueeze(0)
        batch_dict['y'] = self.y[idx].unsqueeze(0) if self.y is not None else None
        batch_dict['fn'] = self.fn[idx] if self.fn is not None else None
        if self.rp is not None:
            batch_dict['rp'] = torch.tensor(self.rp[idx], dtype=torch.long)
        if self.onoff:
            batch_dict['onoff'] = self.onoff[idx] if self.onoff is not None else None
        return batch_dict

def filter_signal(x, cutoff: int or list, mode, sample_rate=250):
    """ filter ecg signal """
    nyq = 125  # sample_rate * 0.5
    xx = x.copy() if isinstance(x, np.ndarray) else x.clone()
    if mode == 'lowpass':
        if cutoff >= nyq: cutoff = nyq - 0.05
        xx = signal.filtfilt(*signal.butter(2, cutoff / nyq, btype='lowpass'), xx, method='gust')
    elif mode == 'highpass':
        xx = signal.filtfilt(*signal.butter(2, cutoff / nyq, btype='highpass'), xx, method='gust')
    elif mode == 'bandpass':
        xx = signal.filtfilt(*signal.butter(2, [cutoff[0] / nyq, cutoff[1] / nyq], btype='bandpass'), xx, method='gust')
    elif mode == 'bandstop':
        xx = signal.filtfilt(*signal.butter(2, [cutoff[0] / nyq, cutoff[1] / nyq], btype='bandstop'), xx, method='gust')
    elif mode == 'notch':
        xx = signal.filtfilt(*signal.iirnotch(cutoff, cutoff, sample_rate), xx, method='gust')
    return xx


class Collate_seq(ABC):
    """
    TBD.

    # usage
        train_loader = torch.utils.data.DataLoader(train_set, batch_size=cfg.run.batch_train, shuffle=True,
                                                   pin_memory=True, collate_fn=Collate_seq(sequence=seq_list))
    """
    def __init__(self, sequence: Union[list, tuple] = tuple([None]), n: int = 10, n_cls: int = 2, th: float = 0.5):
        self.sequence = sequence
        self.n = n
        self.n_cls = n_cls
        self.th = th

    @abstractmethod
    def chunk(self, batch):
        x, y, fn = batch
        x = repeat(x, '(n dim) -> n dim', n=self.n)
        y = reduce(y, '(n dim) -> n', n=self.n, reduction='mean') > self.th
        fn = [fn] * self.n
        return x, y, fn

    @abstractmethod
    def hot(self, batch):
        x, y, fn = batch
        return x, np.squeeze(np.eye(self.n_cls)[y.astype(int).reshape(-1)].transpose()), fn

    @staticmethod
    def sanity_check(batch):
        x, y, fn = batch
        x = torch.tensor(x, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.int32)

        # Guarantee the x and y have 3-dimension shape.
        if len(x.shape) == 1:
            x = x.unsqueeze(0).unsqueeze(0)     # [dim] -> [1, 1, dim]
        elif len(x.shape) == 2:
            x = x.unsqueeze(1)                  # [N, dim] -> [N, 1, dim]
        if len(y.shape) == 1:
            y = y.unsqueeze(1).unsqueeze(1)     # [dim] -> [1, 1, dim]
        elif len(y.shape) == 2:
            y = y.unsqueeze(0)                  # [ch, dim] -> [N, 1, dim]
        return x, y, fn

    def __call__(self, batch):
        x, y, fn = [], [], []
        for b in batch:
            for seq in self.sequence:
                if seq == 'chunk':
                    b = self.chunk(b)
                elif seq == 'hot':
                    b = self.hot(b)
                else:
                    pass
            b = self.sanity_check(b)
            x.append(b[0]), y.append(b[1]), fn.append(b[2])
        return torch.cat(x), torch.cat(y), fn


class BeatCollateSeq:
    def __init__(self, sequence: Union[list, tuple] = tuple([None]), **kwargs):
        self.sequence = sequence
        self.xtype, self.ytype, self.info_type = ['x', 'w', 'onoff'], ['y', 'r_loc'], ['fn', 'rp']
        self.resamp = kwargs['resamp'] if 'resamp' in kwargs.keys() else 64
        self.zit = kwargs['zit'] if 'zit' in kwargs.keys() else True
        self.overlapped = kwargs['overlapped'] if 'overlapped' in kwargs.keys() else 500
        self.fs = 250

    def filter(self, batch):
        nyq = self.fs / 2
        length = batch['x'].shape[-1]
        x = np.concatenate([batch['x'].squeeze()] * 3)
        x = signal.filtfilt(*signal.butter(2, [0.5 / nyq, 50 / nyq], btype='bandpass'), x, method='gust')
        x = signal.filtfilt(*signal.butter(2, [59.9 / nyq, 60.1 / nyq], btype='bandstop'), x, method='gust')
        batch['x'] = torch.tensor(x[length: 2*length].reshape(1, -1).copy(), dtype=torch.float32)
        return batch

    def split(self, batch):
        raw_data = batch['x'].squeeze()
        batch['raw'] = raw_data.tolist()
        remain_length = ((len(raw_data) - 2500) % (2500 - self.overlapped))
        if remain_length != 0:
            raw_data = F.pad(raw_data.unsqueeze(0), (0, 2500 - remain_length), mode='replicate').squeeze()
        splited = raw_data.unfold(dimension=0, size=2500, step=2500-self.overlapped)

        batch['x'] = splited
        batch['fn'] = [batch['fn']] * len(splited)
        return batch

    def r_reg(self, batch):
        try:
            onoff = batch['onoff']
        except KeyError:
            onoff = np.array(sanity_check(label_to_onoff(batch['y'].squeeze()), incomplete_only=True))

        if onoff is None:
            return {}

        try:
            assert len(onoff) == len(batch['rp'])
        except AssertionError:
            # For missing beat or R-gun at inference
            rp = torch.zeros(len(onoff))
            for i_onoff, (on, off, cls) in enumerate(onoff):
                i_on = np.searchsorted(batch['rp'], on)
                i_off = np.searchsorted(batch['rp'], off)
                if i_on+1 == i_off:
                    rp[i_onoff] = batch['rp'][i_on]
            batch['rp'] = rp
        except (TypeError, AttributeError):
            pass
        except KeyError:
            batch['rp'] = None

        raw = batch['x'].squeeze()
        resampled, r_loc = [], []
        for i, (on, off, cls) in enumerate(onoff):
            if sum(np.isnan((on, off))): continue
            if self.zit: on, off = self.zitter(on, off)
            if off >= len(raw)-1: off = -2
            on, off = int(on), int(off)
            chunk = raw[on:off + 1]
            if batch['rp'] is not None:
                r_loc.append((batch['rp'][i] - on) / (off - on))
            resampled.append(torch.tensor(signal.resample(chunk, self.resamp), dtype=torch.float32))

        batch['w'] = torch.stack(resampled, dim=0) if resampled else []
        batch['r_loc'] = torch.tensor(r_loc, dtype=torch.float32) if batch['rp'] is not None else None
        return batch

    def zitter(self, on, off):
        if off - on > 10:
            on += np.random.randint(-3, 4)
            off += np.random.randint(-3, 4)
        else:
            on += np.random.randint(-3, 2)
            off += np.random.randint(-1, 4)
        return max(0, on), off

    def accumulate(self, batches: Union[Tuple, List]):
        accumulate_dict = defaultdict(list)
        # Convert a list of dictionaries per data to a batch dictionary with list-type values.
        for b in batches:
            for k, v in b.items():
                if isinstance(v, list):
                    accumulate_dict[k] += v
                else:
                    accumulate_dict[k].append(v)
        for k, v in accumulate_dict.items():
            try:
                if set(v) == {None}:
                    accumulate_dict[k] = None
                elif k in self.info_type:
                    pass
                elif k in self.xtype:
                    accumulate_dict[k] = torch.cat(v, dim=0).unsqueeze(1) if v else []
                else:
                    accumulate_dict[k] = torch.cat(v, dim=0).squeeze()
            except TypeError: pass
        return accumulate_dict

    def __call__(self, batch: dict or list):
        batches = []
        for b in batch:
            b = self.filter(b)
            for seq in self.sequence:
                if seq == 'r_reg':
                    b = self.r_reg(b)
                elif seq == 'split':
                    b = self.split(b)
                else:
                    pass
            batches.append(b)
        batches = self.accumulate(batches)
        return batches

    def wrap(self, func: Callable = None):
        if func is None: func = self.__call__

        @wraps(func)
        def wrapped_func(*args, **kwargs):
            return func(*args, **kwargs)
        return wrapped_func



class Trainer_stg1(trainer.Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(Trainer_stg1, self).__init__(model, device, mode, optimizer, scheduler)
        self.mini_batch_size = 32

    def train_step(self, batch: dict):
        pass

    def test_step(self, batch: dict):
        x = batch["x"]
        mini_x = x.split(self.mini_batch_size, dim=0)
        out = torch.cat([self.model(m_x) for m_x in mini_x])

        return {'loss': 0, 'output': out}


class Trainer_stg2(trainer.Trainer):
    def __init__(self, model, device, mode: str, optimizer: torch.optim.Optimizer = None, scheduler=None):
        super(Trainer_stg2, self).__init__(model, device, mode, optimizer, scheduler)
        self.mini_batch_size = 128

    def train_step(self, batch: dict):
        pass

    def test_step(self, batch: dict):
        if batch is None: return {'loss': 0, 'output': torch.Tensor([]), 'output_onoff': np.array([])}
        if isinstance((w := batch['w']), torch.Tensor) and isinstance((onoff := batch['onoff']), list):
            mini_w = w.split(self.mini_batch_size, dim=0)
            pred = torch.cat([self.model(m_w) for m_w in mini_w])
            return {'loss': 0, 'output': self.model.clipper(pred), 'output_onoff': onoff}
        else:
            return {'loss': 0, 'output': torch.Tensor([]), 'output_onoff': np.array([])}

