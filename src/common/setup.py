import os
import sys
import json
import random
from collections import namedtuple

import numpy as np
import torch
import torch.backends
import torch.backends.cudnn

from src.common import path


class Setter:
    __cls_var = None

    def __init__(self):
        self.gpu_id, self.SEED, self.device = None, None, None
        SWITCH = int(input('Switch : \n\t1.train\n\t2.test\n\t3.inference\n\tselect >>> '))
        self.SWITCH = SWITCH
        self.set_seed()
        self.set_directory_number()

    def set_seed(self):
        SEED = int(input('Seed number : \n\t >>> '))
        if type(SEED) not in [int, float]: sys.exit("Seed 'number' in [0, 2^32 - 1] please.")
        self.SEED = SEED
        random.seed(SEED)
        np.random.seed(SEED)
        torch.manual_seed(SEED)
        torch.cuda.manual_seed(SEED)
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    def set_directory_number(self):
        num_gpu = torch.cuda.device_count()
        gpu_id = int(input('Select gpu number in [0, %d] for save progress & output.\n\t >>> ' % num_gpu))
        self.gpu_id = gpu_id
        print('\tRunning in device #%d (# of available device: %d)' % (gpu_id, num_gpu))
        device = torch.device(f'cuda:{gpu_id}' if torch.cuda.is_available() else 'cpu')
        torch.cuda.set_device(device)
        self.device = device

    @staticmethod
    def set():
        Setter.__cls_var = Setter()

    @classmethod
    def get(cls):
        if cls.__cls_var is None:
            cls.set()
        return cls.__cls_var


class Configure:
    __cls_var = None

    def __init__(self, filename):
        if not os.path.exists(filename):
            filename = './config/config_example.json'
            print('Cannot find config file. Load file from %s instead...' % filename)
        cfg_dict = json.load(open(filename, 'r'))
        self.cfg_set = self.dict_to_inst(cfg_dict, cls_name='Config_set')

    def dict_to_inst(self, cfg_dict, cls_name: str):
        """
        Convert dictionary to named tuple.
        """
        inst_ = namedtuple(cls_name, sorted(cfg_dict.keys()))
        for k, v in cfg_dict.items():
            if k == 'prefix':
                v = os.path.join('./', v)
            if type(v) is dict:
                cfg_dict[k] = self.dict_to_inst(v, cls_name+'_')
            elif type(v) is str:
                cfg_dict[k] = v
        return inst_(**cfg_dict)

    @classmethod
    def init_cfg(cls):
        cls.__cls_var = None

    @classmethod
    def get_cfg(cls, filename):
        if cls.__cls_var is None:
            cls.__cls_var = cls(filename).cfg_set
        return cls.__cls_var
