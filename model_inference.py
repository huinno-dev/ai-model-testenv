import pandas as pd
import numpy as np
import datetime
import json
import boto3
import pickle
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)
from matplotlib import pyplot as plt
from scipy import signal
import os
from collections import Counter
from src.common.setup import Setter
from scripts import stage_1, stage_2, merged_stage, inference
import time
import copy
from AI_template.AI_common.api import r_peak_detection, sec10_classification


def filter_signal(x, cutoff: int or list, mode, sample_rate=250):
    """ filter ecg signal """
    nyq = 125  # sample_rate * 0.5
    xx = x.copy()
    if mode == 'lowpass':
        if cutoff >= nyq: cutoff = nyq - 0.05
        xx = signal.filtfilt(*signal.butter(2, cutoff / nyq, btype='lowpass'), xx, method='gust')
    elif mode == 'highpass':
        xx = signal.filtfilt(*signal.butter(2, cutoff / nyq, btype='highpass'), xx, method='gust')
    elif mode == 'bandpass':
        xx = signal.filtfilt(*signal.butter(2, [cutoff[0] / nyq, cutoff[1] / nyq], btype='bandpass'), xx, method='gust')
    elif mode in ['notch', 'bandstop']:
        xx = signal.filtfilt(*signal.iirnotch(cutoff, cutoff, sample_rate), xx, method='gust')
    return xx


def notch_filter(Input, fs=250):
    fc = 60  # Cut-off frequency of the filter
    Q = 30  # Quality factor
    b, a = signal.iirnotch(fc, 30, fs)
    filtered_ECG = signal.filtfilt(b, a, Input)
    return filtered_ECG


##############################################################################
print('Access s3')
service_name = 's3'
cfg = json.load(open('config.json'))

s3 = boto3.resource('s3', aws_access_key_id=cfg['access_key'],
                    aws_secret_access_key=cfg['secret_key'])

bucket = s3.Bucket(cfg['bucket_name'])
######################################################################################################################
# Get data from S3
######################################################################################################################
print('Access filenames in s3')
start_idx = datetime.datetime.now()
object_path = 'patch-ecg/'
prefix_objs = bucket.objects.filter(Prefix=object_path)

obj_list = []
for obj in prefix_objs:
    key = obj.key
    obj_list.append(key)
ecg_files = [s for s in obj_list if '.decrypted' in s]

df_path = pd.DataFrame(ecg_files, columns=['path'])
test_id = []
for i in range(0, len(df_path)):
    test_id.append(df_path['path'][i].split('/')[1])
df_path['test_id'] = test_id

end_idx = datetime.datetime.now()
print('Search time')
print(end_idx - start_idx)

##############################################################################
print('Finding target file list')
# t_id = '64acc64e'
t_id = 'a21be44f'

img_path = '/result/img/'

time_list_temp = ['20240304_044930-20240304_045200']
target_time_list = []
for t in range(0, len(time_list_temp)):
    t_ = time_list_temp[t].split('-')
    if len(t_) != 1:
        s_time = datetime.datetime.strptime(t_[0], '%Y%m%d_%H%M%S')
        e_time = datetime.datetime.strptime(t_[1], '%Y%m%d_%H%M%S')
        time_diff = int((e_time - s_time).total_seconds() / 10)
        # Get target time
        for t__ in range(0, time_diff):
            time_target = s_time + datetime.timedelta(seconds=10) * t__
            target_time_list.append(time_target.strftime('%Y%m%d_%H%M%S'))
    else:
        target_time_list.append(t_[0])

data = df_path[df_path['test_id'].str.contains(t_id)].reset_index(drop=True)

target_path = object_path + data['test_id'][0] + '/'
target_objs = bucket.objects.filter(Prefix=target_path)

target_files = []
for obj in target_objs:
    key = obj.key
    target_files.append(key)

# Read target files
print('Downloading ECG')
ecg_file = next((s for s in target_files if 'decrypted' in s), None)

s3_object = s3.Object(cfg['bucket_name'], ecg_file)
object_data = s3_object.get()['Body'].read()

ecg = np.frombuffer(object_data, dtype=np.float16)
start_time = datetime.datetime.strptime('20240219_095700', '%Y%m%d_%H%M%S')

names = []
ecg_list = []
for d in range(0, len(target_time_list)):
    target_time = datetime.datetime.strptime(target_time_list[d], '%Y%m%d_%H%M%S')
    time_diff = (target_time - start_time).total_seconds()
    # Get time
    year = target_time.strftime('%Y')
    month = target_time.strftime('%m')
    day = target_time.strftime('%d')
    hour = target_time.strftime('%H')
    minute = target_time.strftime('%M')
    second = target_time.strftime('%S')
    # Make filename
    name_temp = t_id + '_' + year + month + day + '_' + hour + minute + second
    names.append(name_temp)
    # Get signal
    # 0으로 끝날 때와 아닐 때 구분
    if second[1] == '0':
        s_ = int((time_diff) * 250)
        e_ = int(s_ + 2500)
    else:
        s_ = int((time_diff) * 250)
        e_ = int(s_ + 2500)
    ecg_list.append(np.array(ecg[s_:e_]))

df_temp = pd.DataFrame(names, columns=['filename'])
df_temp['signal'] = ecg_list

p = 0
for p in range(0, len(df_temp)):
    x_raw = df_temp['signal'][p]
    x_2 = np.hstack((x_raw, x_raw))
    x_3 = np.hstack((x_2, x_raw))

    x_notch = notch_filter(x_3)
    x_filtered = filter_signal(x_notch, [0.5, 50], 'bandpass')
    x = x_filtered[2500:5000]

    r_local = r_peak_detection(x, pretrain=False, inference_option='beat')
    beat_local = np.zeros(len(x))
    for r in range(0, len(r_local[0])):
        beat_local[r_local[0][r][0]:r_local[0][r][1]] = r_local[0][r][2] + 2

    r_server = r_peak_detection(x, pretrain=True, inference_option='beat')
    beat_server = np.zeros(len(x))
    for r in range(0, len(r_server[0])):
        beat_server[r_server[0][r][0]:r_server[0][r][1]] = r_server[0][r][2] + 2

    r_pqrst = r_peak_detection(x, pretrain=False, inference_option='pqrst')
    beat_pqrst = np.zeros(len(x))
    for r in range(0, len(r_pqrst[0])):
        beat_pqrst[r_pqrst[0][r][0]:r_pqrst[0][r][1]] = r_pqrst[0][r][2]

    tensec = sec10_classification(x, 16)

    plt.figure(figsize=(20, 4))
    title_txt = df_temp['filename'][p] + \
                ' / 10s model: '+str(tensec[0][0]) + \
                ' / Max, Min' + str(max(x)) + ', ' + str(min(x)) + \
                ' \n 0: Baseline / 1: P / 2: T / 3: N / 4: S / 5: V'

    plt.title(title_txt)
    plt.plot(x, color='black', linewidth=0.7)
    plt.plot(beat_server, color='red', linestyle='dotted', linewidth=0.5)
    plt.plot(beat_local, color='blue', linestyle='dotted', linewidth=0.5)
    plt.plot(beat_pqrst, color='magenta', linestyle='dotted', linewidth=0.5)
    plt.legend(['ECG', 'Beat - server', 'Beat - local', 'PQRST - local'])
    # plt.show()

    plt.savefig(img_path +df_temp['filename'][p] + '.jpg', bbox_inches='tight', pad_inches=0.1, dpi=400)
    plt.close()