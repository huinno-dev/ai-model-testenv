import sys
import copy
from functools import wraps
from typing import Union, Tuple, List, Callable
from collections import defaultdict

import matplotlib.pyplot as plt
from scipy import signal
import pickle as pkl
import numpy as np
from sklearn.model_selection import train_test_split as tts
from sklearn.model_selection import KFold as kf
from sklearn import preprocessing
import torch
from torch.nn import functional as F
from torch.utils.data import Dataset
from src.augmentation import noise_augmentation, pseudo_tachy

from AI_common.operation import label_to_onoff, sanity_check


# --------------------------------- 실전용 --------------------------------- #
def stack_data(data: list or np.ndarray, length=2500, overlapped=500, pad=False, fs=250):
    """
    길이가 안맞는 경우 발생하는 bug 해결.
    """
    copy_data = copy.deepcopy(data)
    if pad:
        last_length = ((len(data) - length) % (length - overlapped))
        if last_length != 0:
            if isinstance(copy_data, list):
                copy_data += [data[-1]] * (length - overlapped - last_length)
            elif isinstance(copy_data, np.ndarray):
                copy_data = np.concatenate((copy_data, [data[-1]] * (length - overlapped - last_length)))
            else:
                raise TypeError

    stacked = []
    for i in range(0, len(copy_data), length - overlapped):
        if i + length <= len(copy_data):
            stacked.append(copy_data[i:i + length])
    return np.array(stacked)


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
        # 반반반반....띵
        n = len(qrs_indexes) // 2
        m1 = recursive_peak_merge(qrs_indexes[:n], n // 2, sampling_rate, overlapped_sec)
        m2 = recursive_peak_merge(qrs_indexes[n:], (len(qrs_indexes) - n) // 2, sampling_rate, overlapped_sec)
        m1 = m1.tolist()
        m2 = m2.tolist()
        m = m1 + m2
        return recursive_peak_merge(m, n, sampling_rate, overlapped_sec)


# --------------------------------- 장난용 --------------------------------- #
def data_preprocess(data, mode='tvt', seed=100, scaling=False, scaler='minmax', filtering=False, cv=False, i_cv=0):
    train_set, valid_set, test_set = None, None, None
    if isinstance(data, tuple):
        for i_d, d in enumerate(data):
            if d is None: continue
            if cv:
                if isinstance(d, list): d = np.array(d)
                tv, test, _ = spliter(d, mode=mode[:-1], seed=seed)
                index_obj = kf(n_splits=5, shuffle=True, random_state=seed).split(tv)
                tr, va = index_obj.__next__()
                for i in range(i_cv):                    tr, va = index_obj.__next__()
                train, valid = tv[tr], tv[va]
            else:
                train, valid, test = spliter(d, mode=mode, seed=seed)

            if train_set is None:
                train_set, valid_set, test_set = train, valid, test
            else:
                train_set = np.concatenate((train_set, train), axis=0)
                if valid_set is not None:
                    valid_set = np.concatenate((valid_set, valid), axis=0)
                if test_set is not None:
                    test_set = np.concatenate((test_set, test), axis=0)
    else:
        if isinstance(data, list): data = np.array(data)
        if cv:
            tv, test, _ = spliter(data, mode=mode[:-1], seed=seed)
            index_obj = kf(n_splits=5, shuffle=True, random_state=seed).split(tv)
            tr, va = index_obj.__next__()
            for i in range(i_cv):                    tr, va = index_obj.__next__()
            train, valid = tv[tr], tv[va]
        else:
            train, valid, test = spliter(data, mode=mode, seed=seed)
        train_set, valid_set, test_set = train, valid, test
    if train_set is None: return train_set, valid_set, test_set

    if filtering:
        filter_fn = lambda x: filter_signal(filter_signal(x, [0.5, 50], 'bandpass'), [59.9, 60.1], 'bandstop')
        train_set = np.array([filter_fn(tr) for tr in train_set])
        if valid_set is not None:
            valid_set = np.array([filter_fn(va) for va in valid_set])
        if test_set is not None:
            test_set = np.array([filter_fn(te) for te in test_set])

    if scaling:
        if scaler == 'minmax':
            train_set = preprocessing.minmax_scale(train_set.squeeze(), axis=1).reshape(-1, 1, 2500)
            if valid_set is not None:
                valid_set = preprocessing.minmax_scale(valid_set.squeeze(), axis=1).reshape(-1, 1, 2500)
            if test_set is not None:
                test_set = preprocessing.minmax_scale(test_set.squeeze(), axis=1).reshape(-1, 1, 2500)
        elif scaler == 'clip':
            train_set = np.expand_dims(np.clip(train_set, -600, 600).reshape(-1, 2500), axis=1)
            if valid_set is not None:
                valid_set = np.expand_dims(np.clip(valid_set, -600, 600).reshape(-1, 2500), axis=1)
            if test_set is not None:
                test_set = np.expand_dims(np.clip(test_set, -600, 600).reshape(-1, 2500), axis=1)
        else:
            mag = scaler if isinstance(scaler, int) else 1
            train_set = np.expand_dims(mag * train_set, axis=1)
            if valid_set is not None:
                valid_set = np.expand_dims(mag * valid_set, axis=1)
            if test_set is not None:
                test_set = np.expand_dims(mag * test_set, axis=1)

    return train_set, valid_set, test_set


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


def spliter(data, mode='tvt', seed=100):
    if len(mode) == 1:
        return data, None, None
    else:
        ab, c = tts(data, test_size=0.2, random_state=seed)
        if len(mode) == 2:
            return ab, c, None
        elif len(mode) == 3:
            a, b = tts(ab, test_size=0.2, random_state=seed)
            return a, b, c
        else:
            sys.exit('Unexpected mode')


# --------------------------------- Dataset --------------------------------- #
class ECG_dataset(Dataset):
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.x = torch.tensor(kwargs['x'], dtype=torch.float32)
        self.y = torch.tensor(np.array(kwargs['y']), dtype=torch.long) if self.attr('y') is not None else None
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
        # 파일명 출력
        print(idx)
        print(self.fn[idx])
        print(np.shape(self.x))
        print(np.shape(self.y))
        # 길이 오버할 경우
        if idx >= len(self.y):
            raise IndexError(f"Index {idx} is out of bounds for dataset of size {len(self.y)}")

        batch_dict['x'] = self.x[idx].unsqueeze(0)
        batch_dict['y'] = self.y[idx].unsqueeze(0) if self.y is not None else None
        batch_dict['fn'] = self.fn[idx] if self.fn is not None else None
        if self.rp is not None:
            batch_dict['rp'] = torch.tensor(self.rp[idx], dtype=torch.long)
        if self.onoff is not None:
            batch_dict['onoff'] = self.onoff[idx]
        return batch_dict


def peak_to_label(peak_list: Union[np.ndarray, List[int]],
                  beat_type: Union[np.ndarray, List] = None,
                  length: int = 2500, arm: int = 15) -> np.ndarray:
    if beat_type is None:
        beat_type = [None] * len(peak_list)
    else:
        assert len(peak_list) == len(beat_type)

    sets = ['N', 'S', 'V', 'Q']
    beat_type = [sets.index(b)+1 for b in beat_type]

    canvas = np.zeros(length, dtype=np.long)
    for p, t in zip(peak_list, beat_type):
        canvas[max(0, (p-arm+1)): min(length, p+arm)] = 1 if t is None else t
    return canvas


class Collate_seq:
    def __init__(self, sequence: Union[list, tuple] = tuple([None]), **kwargs):
        self.sequence = sequence
        self.xtype, self.ytype, self.info_type = ['x', 'w', 'onoff'], ['y', 'r_loc'], ['fn', 'rp']
        self.resamp = kwargs['resamp'] if 'resamp' in kwargs.keys() else 64
        self.zit = kwargs['zit'] if 'zit' in kwargs.keys() else True
        self.overlapped = kwargs['overlapped'] if 'overlapped' in kwargs.keys() else 500
        self.filter_fn = lambda x: filter_signal(filter_signal(x, [0.5, 50], 'bandpass'), [59.9, 60.1], 'bandstop')

        if 'noise_aug' in sequence:
            self.emg_list = []
            for emg_fn in ['./src/resource/EMG1.pkl', './src/resource/EMG2.pkl']:
                with open(emg_fn, 'rb') as fp: self.emg_list.append(self.filter_fn(pkl.load(fp)))

    def split(self, batch):
        raw_data = batch['x'].squeeze()
        raw_y = batch['y'].squeeze().float() if batch['y'] is not None else None
        batch['raw'] = raw_data.tolist()
        remain_length = ((len(raw_data) - 2500) % (2500 - self.overlapped))
        if remain_length != 0:
            raw_data = F.pad(raw_data.unsqueeze(0), (0, 2500 - remain_length), mode='replicate').squeeze()
            if raw_y is not None:
                raw_y = F.pad(raw_y.unsqueeze(0), (0, 2500 - remain_length), mode='replicate').squeeze()
        splited = raw_data.unfold(dimension=0, size=2500, step=2500 - self.overlapped)
        splited_y = raw_y.unfold(dimension=0, size=2500, step=2500 - self.overlapped) if raw_y is not None else None

        batch['x'] = splited
        batch['y'] = splited_y.long()
        batch['fn'] = [batch['fn']] * len(splited)
        return batch

    def noise_augment(self, batch):
        augmented = noise_augmentation(batch['x'].clone(), self.emg_list).clone()
        augmented = filter_signal(filter_signal(augmented, [0.5, 50], 'bandpass'), [59.9, 60.1], 'bandstop')
        batch['x'] = torch.tensor(augmented.copy(), dtype=torch.float32)
        return batch

    def pseudo_tachy(self, batch):
        if torch.rand(1) < 0.15:
            batch['x'], batch['y'] = pseudo_tachy(batch['x'], batch['y'], batch['rp'])
        return batch

    def r_reg(self, batch):
        try:
            onoff = batch['onoff']
        except KeyError:
            onoff = np.array(sanity_check(label_to_onoff(batch['y'].squeeze()), incomplete_only=True))

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

        batch['w'] = torch.stack(resampled, dim=0)
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
                    cat = torch.cat(v, dim=0).unsqueeze(1)
                    accumulate_dict[k] = cat.reshape(cat.shape[0], -1, 2500)
                else:
                    accumulate_dict[k] = torch.cat(v, dim=0).squeeze(1)
            except TypeError: pass
        return accumulate_dict

    def __call__(self, batch: dict or list):
        batches = []
        for b in batch:
            for seq in self.sequence:
                if seq == 'r_reg':
                    b = self.r_reg(b)
                elif seq == 'split':
                    b = self.split(b)
                elif seq == 'noise_aug':
                    b = self.noise_augment(b)
                elif seq == 'p_tachy':
                    b = self.pseudo_tachy(b)
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


def combine_label(labels, mode: list or tuple):
    labels = np.copy(labels)
    if 'rhythm' in mode:
        labels[np.where(labels == 3)] = 2
    elif 'beat' in mode:
        labels[np.where(labels == 3)] = 1
    elif 'weird' in mode:
        labels[np.where(labels == 2)] = 1
        labels[np.where(labels == 3)] = 2
    elif 'all' in mode:
        labels[np.where(labels == 2)] = 1
        labels[np.where(labels == 3)] = 1
    return labels


def remove_data(data, label, criterion):
    data, label = np.array(data), np.array(label)

    check_idx = (label == criterion).sum(1)
    rid_idx = check_idx == 0

    return data[rid_idx], label[rid_idx]
