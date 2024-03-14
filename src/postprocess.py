import sys
import numpy as np
import cv2
import matplotlib.pyplot as plt
from collections import defaultdict, Counter

from src.loss import get_IOU


def label_to_wave(value, modulo: int = 10):
    if value % modulo == 1:
        return 'P'
    elif value % modulo == 2:
        return 'W'
    elif value % modulo == 3:
        return 'T'


def label_to_Dx(value, tens: int = 10):
    if value // tens == 0:
        return ''
    elif value // tens == 1:
        return '_apc'
    elif value // tens == 2:
        return '_vpc'


def fetch_class_multi_to_binary(data: np.ndarray, num_label: int, baseline_idx: int = 0):
    fetched = []
    for i_cls in range(num_label):
        if i_cls != baseline_idx:
            tmp_arr = data.copy()
            tmp_arr[tmp_arr != i_cls] = baseline_idx
            fetched.append(tmp_arr)

    return tuple(fetched)


def smoothing(data: tuple, operation: str = 'CO', kernel_size: tuple = (20, 20)):
    if len(operation) != len(kernel_size):
        sys.exit('Length of operation and kernel_size must be same.')

    morphed = []
    for d in data:
        tmp_d = np.hstack((np.zeros((d.shape[0], 100)), d, np.zeros((d.shape[0], 100))))
        for o, k in zip(operation.lower(), kernel_size):
            kernel = np.ones((1, k))
            if o == 'c':
                tmp_d = cv2.morphologyEx(tmp_d, cv2.MORPH_CLOSE, kernel)
            elif o == 'o':
                tmp_d = cv2.morphologyEx(tmp_d, cv2.MORPH_OPEN, kernel)
            else:
                continue
        morphed.append(tmp_d[:, 100:-100])
    return tuple(morphed)


def accumulate_morphed(morphed: tuple):
    morphed_clip = np.clip(np.array(morphed), 0, 1)
    morphed_sum = np.sum(morphed, axis=0)
    morphed_sum_clip = np.sum(morphed_clip, axis=0)

    on_off_clip = np.array(label_to_onoff(morphed_sum_clip), dtype=np.int32)
    overlapped_group_idx = np.where(on_off_clip[:, 2] > 1)[0]

    # If 2 or more labels overlapped via morphological operation,
    # overlapped range is distributed into pre- and post- label based on the center point.
    if len(overlapped_group_idx) != 0:
        on_off = np.array(label_to_onoff(morphed_sum))
        for i, idx in enumerate(overlapped_group_idx):
            pre_group_off_idx = on_off_clip[idx - 1][1]
            post_group_on_idx = on_off_clip[idx + 1][0]  # end_pnt로 되어있는 부분 수정

            off_label = on_off[np.where(on_off == pre_group_off_idx)[0][0]][-1]
            on_label = on_off[np.where(on_off == post_group_on_idx)[0][0]][-1]

            overlapped_range = on_off_clip[np.where(on_off_clip[:, 2] > 1)[0]][i][:2]
            center = int(sum(overlapped_range) // len(overlapped_range))
            morphed_sum[overlapped_range[0] - 1:center] = off_label
            morphed_sum[center:overlapped_range[1] + 1] = on_label

    return morphed_sum


def accumulate_morphed_10sec(morphed: tuple):
    if len(np.array(morphed).shape) < 3:
        return morphed
    else:
        accumulated = morphed[0]
        for i_m, m in enumerate(morphed):
            if i_m == 0: continue
            for i in range(1, len(accumulated)):
                empty_idx = accumulated[i] == 0
                accumulated[i] += m[i] * empty_idx

    return accumulated


def ballot_counting(ballots, th_iou=0.5, th_vote=3):
    """
    Inputs:
        ballots - List (or np.ndarray) of prediction from N models.
        th_iou - IoU threshold for determining whether the predictions of different N models are the same.
        th_vote - counting threshold to aggregate predictions from different N models.
    Output:
        Aggregation of N predictions.
    Docs:
        Receive N predictions from different N models as list (or np.ndarray).
        Each prediction is converted to onoff and registered in the dictionary as a (key-value) pair of ('on'-[on off]).
        For new onoff,
            1. Compare 'key' and 'on' to find the closest onoff in the dictionary.
            2. Calculate IoU using 'value' and 'on~off'.
            3. If an 'item' satisfying th_iou exists in the dictionary, onoff is added to the item.
                Otherwise, onoff is registered as a new entry in the dictionary.
        Nested function:
            Receive onoff_list and return averaged on & off.
    """
    def avg_oo(onoff_list, astype: type = None):
        if len(onoff_list) == 0:
            return None, None
        on__, off__, cls__ = 0, 0, []
        for c in onoff_list:
            on__ += c[0] / len(onoff_list)
            off__ += c[1] / len(onoff_list)
            cls__.append(c[-1])
        most_cls = Counter(cls__).most_common()[0][0]
        if astype:
            on__, off__ = astype(on__), astype(off__)
        return on__, off__, most_cls

    candidate = defaultdict(list)
    for i, ballot in enumerate(ballots):
        onoff = list(sanity_check(label_to_onoff(np.array(ballot).squeeze(), middle_only=True), incomplete_only=True))
        if i == 0:
            for oo in onoff:
                candidate[oo[0]].append(list(oo))
            continue

        for oo in onoff:
            idx = np.searchsorted(list(candidate.keys()), oo[0])        # find nearest key in candidate.
            if idx == 0:
                pass                    # if 'on' is less than first 'key'.
            elif idx == len(candidate):
                idx -= 1                # if 'on' is more than last 'key'.
            else:
                left, right = list(candidate.keys())[idx-1:idx+1]
                if oo[0] - left < right - oo[0]:
                    idx -= 1            # 'on' is closer to left than right

            candi_k, candi_v = list(candidate.items())[idx]
            iou = get_IOU(avg_oo(candi_v)[:-1], (oo[0], oo[1]))
            if iou >= th_iou:
                candidate[candi_k].append(oo)       # append to exist item
            else:
                candidate[oo[0]].append(oo)         # register in the dictionary as new item
    result = np.array([avg_oo(it[1], astype=int) for it in candidate.items() if len(it[1]) >= th_vote])
    result_array = onoff_to_label(np.sort(result, axis=0), num_pat=1).squeeze()      # sort onoff
    return result_array


def Dx_naming(label):
    mapping_dict = {3: 22, 2: 12, 1: 2}         # Reverse order to avoid conflict

    label_ = label
    for key in mapping_dict.keys():
        label_[label_ == key] = mapping_dict[key]

    on_off = label_to_onoff(label_)

    # naming (label -> Dx)
    txt_list = []

    for idx in range(len(on_off)):
        off_pre, on_post = 20000, 20000
        wave_pre, wave_post = '', ''
        on, off, label = on_off[idx][:]

        if idx != 0:
            off_pre, label_pre = on_off[idx - 1][[1, 2]]
            wave_pre = label_to_wave(label_pre)
        elif idx != len(on_off) - 1:
            on_post, label_post = on_off[idx + 1][[0, 2]]
            wave_post = label_to_wave(label_post)

        wave = label_to_wave(label)
        Dx = label_to_Dx(label)

        txt_on = f'{on}, {wave}_on{Dx}'
        if on != 20000:
            if (on - off_pre != 1) or ([wave_pre, wave] not in [['T', 'P'], ['W', 'T'], ['T', 'W']]):
                txt_list.append(txt_on)

        txt_off = f'{off}, {wave}_off{Dx}'
        if off != 20000:
            if (on_post - off != 1) or [wave, wave_post] not in [['T', 'P'], ['W', 'T'], ['T', 'W']]:
                txt_list.append(txt_off)
            elif (on_post - off == 1) and [wave, wave_post] in [['T', 'P'], ['W', 'T'], ['T', 'W']]:
                txt_list.append(f'{off}, {wave}_{wave_post} 경계')

    return txt_list


if __name__ == '__main__':
    cls = 1
    tmp = [
        [[10, 20], [30, 40], [50, 60]],
        [[11, 21], [32, 42], [52, 62]],
        [[8, 22], [29, 43], [47, 60]],
        [[10, 20], [24, 28], [50, 60]]  # 1 missing, 1 r-gun
    ]
    ttmp = [onoff_to_label(np.concatenate((np.array(t), np.ones((len(t), 1))), axis=1), 100) for t in tmp]
    tttmp = ballot_counting(ttmp)
    print(tttmp)
