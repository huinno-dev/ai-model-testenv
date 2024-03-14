import os
import warnings
from typing import List, Union
from collections import defaultdict
import math
import wfdb
import pickle as pkl
import numpy as np
import torch
from scipy import signal as sig

from src.common.decryption import bitwise_operation, decryption
from src.data import data_preprocess, filter_signal, peak_to_label
from src.computer import remove_tachy
from AI_template.AI_common.operation import onoff_to_label

warnings.filterwarnings("ignore", category=np.VisibleDeprecationWarning)


class DatasetParser_v2:
    """
    Load data and preprocessing.

    :param data_directory: List[str]
    :seed: int.
        Used to split train-valid dataset with filename criterion.
    """
    def __init__(self, data_directory: Union[List[str], str] = '', seed: int = 0):
        self.data_directory = defaultdict(list)
        self.seed = seed
        self.check_data_path(data_directory)

        self.db_dict = {
            'mit-normal': {'path': "mit-bih-normal-sinus-rhythm-database-1.0.0/", 'ch': 'ECG1', 'ext': 'atr'},
            'mit-arrhythmia': {'path': "mit-bih-arrhythmia-database-1.0.0/", 'ch': 'MLII', 'ext': 'atr'},
            'mit-noise': {'path': "mit-bih-noise-stress-test-database-1.0.0/", 'ch': 'MLII', 'ext': 'atr'},
            'mit-long': {'path': "mit-bih-long-term-ecg-database-1.0.0", 'ch': 'ECG1', 'ext': 'atr'},
            'mit-svt': {'path': "mit-bih-supraventricular-arrhythmia-database-1.0.0", 'ch': 'ECG1', 'ext': 'atr'},
            'ludb': {'path': "lobachevsky-university-electrocardiography-database-1.0.1/data", 'ch': 'i', 'ext': 'ii'},
            'huinno': {'path': "huinno/entire", 'ch': None, 'ext': 'npy'},
            'ltafdb': {'path': "long-term-af-database-1.0.0", 'ch': 0, 'ext': 'atr'},
            'chfdb': {'path': "bidmc-congestive-heart-failure-database-1.0.0", 'ch': 'ECG1', 'ext': 'ecg'},
            'aha': {'path': "ahadb", 'ch': 'MLII', 'ext': 'atr'}
        }

    def check_data_path(self, data_directory, extension='.pkl'):
        if isinstance(data_directory, list):
            for data_direc in data_directory: self.check_data_path(data_direc)
        else:
            train_path = os.path.join(data_directory, 'train')

            if os.path.isdir(train_path):
                tv_list = [os.path.join(train_path, l) for l in os.listdir(train_path) if l.endswith(extension)]
                train_list, valid_list, _ = data_preprocess(tv_list, mode='tv', seed=self.seed)
                self.data_directory['train'] += train_list.tolist()
                self.data_directory['valid'] += valid_list.tolist()
            else:
                print(f'Cannot find directory {train_path}')

            test_path = os.path.join(data_directory, 'test')
            if os.path.isdir(test_path):
                self.data_directory['test'] += [
                    os.path.join(test_path, l) for l in os.listdir(test_path) if l.endswith('.pkl')]
            else:
                print(f'Cannot find directory {test_path}')
            self.data_directory['entire'] = \
                self.data_directory['train'] + self.data_directory['valid'] + self.data_directory['test']

    # parse data from pkl data (hospital or 10 sec chunked patch data)
    def parse(self, mode, long=False):
        """
        Parse the data appropriate to mode.
        :mode: str.

        :return: Dictionary.
        The data is parsed from appropriate filenames and accumulated in the dictionary.
        Each key & value corresponds to the single patch from the raw data.
        """
        path_list = self.data_directory[mode] if mode in self.data_directory.keys() else None
        if path_list is None: return {}
        recordings, labels, filenames, rpeaks = [], [], [], []
        # 데이터 오류 발생 시 다음 데이터로 넘어가도록 설정
        try:
            for f_path in path_list:
                with open(f_path, 'rb') as p:
                    df = pkl.load(p)

                for i in range(0, len(df)):
                    try:
                        recordings.append(df['signal_long' if long else 'signal'][i])
                        signal = df['signal_long' if long else 'signal'][i]
                    except KeyError:
                        recordings.append(df['signal'][i])
                        signal = df['signal'][i]

                    # Append required variables
                    out = df['outside_idx'][i]
                    filenames.append(df['filename'][i])

                    # R peak
                    label_beat, on, off = df['label_beat'][i] + 3, df['QRS'][i][:, 0], df['QRS'][i][:, 1]
                    rp = df['r_idx'][i]
                    # if first qrs_on is nan
                    if math.isnan(on[0]):
                        # case1: first r_idx is before qrs_off -> change qrs_on to 0
                        if rp[0] < off[0]:
                            on[0] = 0
                        # case2: first r_idx is after qrs_off -> drop first qrs pair
                        elif rp[0] > off[0]:
                            on = on[1:]
                            off = off[1:]
                    # if first qrs_off is nan
                    if math.isnan(off[-1]):
                        # case1: last r_idx is after qrs_on -> change qrs_off to len(signal)-1
                        if rp[-1] > on[-1]:
                            off[-1] = len(signal) - 1
                        # case2: last r_idx is before qrs_on -> drop last qrs pair
                        elif rp[-1] < on[-1]:
                            on = on[:-1]
                            off = off[:-1]
                    rpeaks.append(rp)
                    onoff = torch.tensor(np.stack((on, off, label_beat), axis=1))

                    # beat segmentation labels
                    # 0 -> baseline
                    # 1 -> P
                    # 2 -> T
                    # 3, 4, 5 -> N, S, V
                    label = np.zeros(len(signal))
                    # R
                    if (len(label_beat) == len(on)) & (len(label_beat) == len(off)):
                        for l in range(0, len(label_beat)):
                            s = int(on[l])
                            e = int(off[l])
                            label[s:e] = label_beat[l]
                    else:
                        print('------------- QRS and R length mismath -------------')
                    # each segment
                    segments = ['P', 'T']
                    for seg in segments:
                        # if target segment is zero -> pass to next segment
                        if len(df[seg][i]) == 0:
                            pass
                        else:
                            on_seg = df[seg][i][:, 0]
                            off_seg = df[seg][i][:, 1]
                            # get range to update
                            for l in range(0, len(df[seg][i])):
                                try:
                                    s = int(on_seg[l])
                                    e = int(off_seg[l])
                                # np.Nan -> 0, 2499
                                except ValueError:
                                    # On
                                    if (type(s) != int):
                                        s = int(0)
                                        e = int(off_seg[l])
                                    # off
                                    elif (type(e) != int):
                                        s = int(on_seg[l])
                                        e = int(len(label) - 1)
                                if seg == 'P':
                                    label[s:e] = 1
                                elif seg == 'T':
                                    label[s:e] = 2
                                else:
                                    label[s:e] = 9
                    labels.append(label)
        except:
            pass
        return {'x': np.array(recordings), 'y': np.array(labels), 'fn': filenames, 'rp': rpeaks}

    @staticmethod
    #
    def parse_from_path(path):
        with open(path, 'rb') as p:
            df = pkl.load(p)

        recordings, labels, filenames, rpeaks = [], [], [], []
        for i in range(len(df)):
            recordings.append(df['signal'][i])
            out = df['outside_idx'][i]
            label_beat, on, off = df['label_beat'][i] + 1, df['w_on'][i], df['w_off'][i]
            if on[0] == out: on[0] = 0
            if off[-1] == out: off[-1] = 2499
            onoff = torch.tensor(np.stack((on, off, label_beat), axis=1))
            labels.append(onoff_to_label(onoff))
            filenames.append(df['filename'][i])

            rp = df['r_idx'][i]
            if rp[0] > onoff[0, 1]: rp.insert(0, out)
            if rp[-1] < onoff[-1, 0]:   rp.append(out)
            rpeaks.append(rp)
        return {'x': np.array(recordings), 'y': np.array(labels), 'fn': filenames, 'rp': rpeaks}

    def read_db(self, db_name, sec=60):
        db_root = 'E:/database/openDB'
        db_dict = self.db_dict[db_name]
        db_path = os.path.join(db_root, db_dict['path'])

        anno_list = ['N', 'L', 'R', 'A', 'a', 'J', 'S', 'V', 'F', 'e', 'j', 'E', '/', 'f', 'Q', '!']
        convert_dict = {'N': ['N', 'L', 'R', 'f', 'F'], 'S': ['S', 'a', 'A', 'J'], 'V': ['V', '!'],
                        'Q': ['Q', 'e', 'j', 'E', '/']}
        convert_dict = {i: k for k, v in convert_dict.items() for i in v}

        resample_fn = lambda x, s, fs: np.array(sig.resample(x.squeeze()[:s], int(s * 250 / fs)))
        filter_fn = lambda x: filter_signal(filter_signal(x, [0.5, 50], 'bandpass'), [59.9, 60.1], 'bandstop')

        ecgs, labels, rpeaks, fns = [], [], [], []
        print(f'Load {db_name} dataset...')

        if db_name == 'huinno':
            ecgs, rpeaks, fns = huinno_db(db_path, sec)
            return {'x': np.array(ecgs), 'y': None, 'fn': fns, 'rp': rpeaks}

        for fn in os.listdir(db_path):
            if fn.endswith('.dat'):
                fn_ = fn.replace('.dat', '')
                # Get signal
                if isinstance(db_dict['ch'], str):
                    data, info = wfdb.rdsamp(os.path.join(db_path, fn_), channel_names=[db_dict['ch']])
                    if (info['sig_name'] is None) or (db_dict['ch'] not in info['sig_name']): continue
                elif isinstance(db_dict['ch'], int):
                    data, info = wfdb.rdsamp(os.path.join(db_path, fn_))
                    data = data[:, db_dict['ch']]
                else:
                    raise TypeError('Unexpected type.')
                if np.isnan(data).sum(): continue
                print(f"\t---- {fn_}")
                if info['units'][0].lower() != 'mv':
                    print('\t\t', info['units'][0])
                strip = len(data) if sec is None else min(info['fs'] * sec, len(data))
                filtered = filter_fn(resample_fn(data, strip, info['fs']))
                anno = wfdb.rdann(os.path.join(db_path, fn_), extension=db_dict['ext'])
                finded = np.in1d(anno.symbol, anno_list)
                types = [convert_dict[t] for t in np.array(anno.symbol)[finded].tolist()]
                anno = np.unique(anno.sample[finded])
                anno = np.array(anno[anno <= strip] * 250 / info['fs'], dtype=np.int)
                p_y = peak_to_label(anno, beat_type=types, length=len(filtered), arm=15)
                p_y[p_y == 4] = 1
                ecgs.append(filtered), labels.append(p_y.reshape(1, -1)), rpeaks.append(anno), fns.append(fn_)
        return {'x': np.array(ecgs), 'y': labels, 'fn': fns, 'rp': rpeaks}


def huinno_db(path, sec):
    ecgs, rpeaks, fns = [], [], []
    files = os.listdir(path)
    for f in files:
        if f.endswith('.txt'):
            print(f"\t---- {f}")
            signal = np.fromstring(open(os.path.join(path, f), 'r').read(), dtype=float, sep='\n').squeeze()
            strip = len(signal) if sec is None else min(250 * sec, len(signal))
            resampled_data = np.array(sig.resample(signal.squeeze()[:strip], strip))
            filtered = filter_signal(filter_signal(resampled_data, [0.5, 50], 'bandpass'), [59.9, 60.1], 'bandstop')
            peaks = np.load(os.path.join(path, f).replace('.txt', '_annotation.npy'))
            peaks = remove_tachy(np.array(sorted(set(peaks))))
            ecgs.append(filtered)
            rpeaks.append(peaks)
            fns.append(f[:-4])
    return ecgs, rpeaks, fns


def save_pkl_data(saving_data, name: str):
    with open(name, 'wb') as p:
        pkl.dump(saving_data, p)


def read_from_txt(path):
    data_list, filenames = [], []
    if isinstance(path, list):
        for p in path:
            data_list.append(np.loadtxt(p))
            filenames.append(os.path.split(p)[-1])
    elif isinstance(path, str):
        if os.path.isdir(path):
            for f_name in os.listdir(path):
                if f_name.endswith('.txt'):
                    data_list.append(np.loadtxt(os.path.join(path, f_name)))
                    filenames.append(f_name)
        else:
            return np.loadtxt(path), os.path.split(path)

    return np.array(data_list), np.array(filenames)


def read_bin_file(f_name):
    serial_idx = f_name.find('_output')
    serial = f_name[serial_idx - 5:serial_idx] if serial_idx else None
    # serial = '07'
    with open(f_name, "rb") as tmp:
        bin_data = tmp.read()
    # if head of data includes device serial, remove header and take body only.
    HEADER = True if serial in str(bin_data[:32]) else False

    if HEADER:
        header = bin_data[:32]
        # Get data & unix time
        start_unix_time = int.from_bytes(header[13:18], "big")
        event_num = int.from_bytes(header[18:20], 'big')
        event_size = int(header[20])
        timestamp_num = int.from_bytes(header[21:24], 'big')
        page_size = int.from_bytes(header[24:26], 'big')
        if page_size == 2048:
            page_size = 2040
        elif page_size == 4096:
            page_size = 4080
        else:
            raise ValueError
        pos = 32

        # User marking events
        events = []
        for i in range(event_num):
            event = bin_data[pos:pos + event_size]
            event_unix_time = int.from_bytes(event[:5], 'little')
            event_type = int.from_bytes(event[-4:], 'little')
            events.append((event_unix_time, event_type))
            pos += event_size

        # Decryption
        a = bytes([0xd6, 0x73, 0xd2, 0x06, 0xa1, 0x87, 0x59, 0xf6, 0x69, 0x7f, 0xb5, 0xd7, 0x10, 0xb2, 0xb4, 0x4d])
        p = bytes([0x61, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff, 0xff])
        patch_public_key_size = 16
        patch_public_key = bin_data[pos:pos + patch_public_key_size]
        pos += patch_public_key_size

        # Raw data
        cipher_text = bin_data[pos:pos + timestamp_num * page_size]
        # data = cipher_text
        k = pow(int.from_bytes(patch_public_key, 'little'), int.from_bytes(a, 'little'), int.from_bytes(p, 'little'))
        kb = k.to_bytes(16, 'little')
        bin_data = decryption(kb, cipher_text)
    else:
        serial_candi = os.path.split(f_name)[-1]
        find_idx = serial_candi.find('_raw')
        if find_idx:
            serial = serial_candi[:find_idx]

    bin_data = np.frombuffer(bin_data, dtype=np.uint8).tolist()
    print('bin_file: ', serial)
    return bitwise_operation(bin_data), serial


def get_holter_analysis(path, f_name, hz=250):
    beat_list = ['N', 'V', 'S', 'U']
    full_name = os.path.join(path, f_name + '.txt')
    lines = open(full_name, 'r').readlines()
    arr = np.array([line[:-1].split('\t') for line in lines[1:]])
    idx_list = [i for i, a in enumerate(arr) if a[1] in beat_list]
    times = np.array(arr[idx_list, 0], dtype=np.float)
    return np.array(times * hz, dtype=np.int), np.array(arr[idx_list, -1])


