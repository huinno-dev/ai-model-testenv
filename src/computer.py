import numpy as np


class R_Count:
    def __init__(self, tol: int = 9):
        self.TOL = tol
        self.log_dict = {}
        self.clear()

    def clear(self):
        self.log_dict = {
            'n_peaks': 0,
            'correct': 0,
            'miss': 0,
            'chop': 0,
            'r_gun': 0
        }

    def cal(self, r_indices, prediction):
        self.log_dict['n_peaks'] += len(r_indices)
        miss, correct, chop = 0, 0, 0
        for index in r_indices:
            l_gap = np.searchsorted(prediction, index - self.TOL, side='left')
            r_gap = np.searchsorted(prediction, index + self.TOL, side='right')
            dot = r_gap - l_gap
            if dot > 1:
                chop += (dot - 1)
                correct += 1
            elif dot == 1:
                correct += 1
            else:
                miss += 1
        self.log_dict['correct'] += correct
        self.log_dict['chop'] += chop
        self.log_dict['miss'] += miss
        self.log_dict['r_gun'] += len(prediction) - correct - chop

    def logging(self):
        print(f'R count with Tolerance {self.TOL}')
        txt = ''
        for k, v in self.log_dict.items():
            txt += f'\t{k.upper()}:\t{v}\n'
        txt += f"\tAcc:\t{100 * self.log_dict['correct'] / (self.log_dict['n_peaks'] + self.log_dict['r_gun']):.4f}%"
        print(txt)


def r_count(r_indices, prediction, TOL=3):
    n_peaks = len(r_indices)
    miss, correct, chop = 0, 0, 0
    for index in r_indices:
        l_gap = np.searchsorted(prediction, index - TOL, side='left')
        r_gap = np.searchsorted(prediction, index + TOL, side='right')
        dot = r_gap - l_gap
        if dot > 1:
            chop += (dot-1)
            correct += 1
        elif dot == 1:            correct += 1
        else:            miss += 1
    r_gun = len(prediction) - correct - chop
    return print('R count with Tolerance %d'
                 '\n\tTotal:\t%d\n\tCorrect:\t%d\n\tMiss:\t%d\n\tChop:\t%d\n\tR-Gun:\t%d\n\tAcc:\t%f'
                 % (TOL, n_peaks, correct, miss, chop, r_gun, correct/(n_peaks+r_gun)))


def remove_tachy(r_peak_indices, max_hr=220, fs=250):
    indices = np.copy(r_peak_indices)

    hr_peak_to_peak = fs / np.diff(indices) * 60

    # Wrong peak (peek to peek hr > 220) removal
    hr_peak_to_peak_index = np.where(hr_peak_to_peak > max_hr)[0]
    i = 0
    while i < len(hr_peak_to_peak_index):
        if hr_peak_to_peak[hr_peak_to_peak_index[i]] < max_hr:
            hr_peak_to_peak_index = hr_peak_to_peak_index[1:]
            continue
        prev_hr_peak_to_peak = hr_peak_to_peak[hr_peak_to_peak_index[i] - 1]
        cur_hr_peak_to_peak = hr_peak_to_peak[hr_peak_to_peak_index[i]]

        if hr_peak_to_peak_index[i] + 1 < len(hr_peak_to_peak):
            next_hr_peak_to_peak = hr_peak_to_peak[hr_peak_to_peak_index[i] + 1]
        else:
            indices = np.delete(indices, -1)
            hr_peak_to_peak = np.delete(hr_peak_to_peak, -1)
            hr_peak_to_peak_index = np.delete(hr_peak_to_peak_index, -1)
            continue

        new_hr_1_1 = (cur_hr_peak_to_peak * prev_hr_peak_to_peak) / (cur_hr_peak_to_peak + prev_hr_peak_to_peak)
        new_hr_1_2 = next_hr_peak_to_peak
        new_hr_2_1 = prev_hr_peak_to_peak
        new_hr_2_2 = (cur_hr_peak_to_peak * next_hr_peak_to_peak) / (cur_hr_peak_to_peak + next_hr_peak_to_peak)

        if abs(new_hr_1_1 - new_hr_1_2) < abs(new_hr_2_1 - new_hr_2_2):
            indices = np.delete(indices, hr_peak_to_peak_index[i])
            hr_peak_to_peak[hr_peak_to_peak_index[i] - 1] = new_hr_1_1
            hr_peak_to_peak = np.delete(hr_peak_to_peak, hr_peak_to_peak_index[i])
            if new_hr_1_1 <= max_hr:
                hr_peak_to_peak_index = np.delete(hr_peak_to_peak_index, i)
            hr_peak_to_peak_index -= 1
        else:
            indices = np.delete(indices, hr_peak_to_peak_index[i] + 1)
            hr_peak_to_peak[hr_peak_to_peak_index[i]] = new_hr_2_2
            hr_peak_to_peak = np.delete(hr_peak_to_peak, hr_peak_to_peak_index[i] + 1)
            if new_hr_2_2 <= max_hr:
                hr_peak_to_peak_index = np.delete(hr_peak_to_peak_index, i)
            hr_peak_to_peak_index -= 1
    return indices


def idx_to_onoff(indices: np.ndarray, sense) -> list:
    cnt, pre, on = 1, -2, -2
    groups = []
    if not indices.shape: return groups
    for idx in indices:
        # continuous
        if idx == pre+1:
            cnt += 1
            pre = idx
        # not continuous
        else:
            if cnt >= sense:    groups.append([on, pre])
            cnt = 1
            on = pre = idx  # new buffer
    # last corner case
    if cnt >= sense:    groups.append([on, pre])
    return groups


def unit_slicing(arr: np.array, idx, crit_axis=0):
    if len(arr.squeeze().shape) == 1: arr = np.expand_dims(arr, axis=1)

    tmp_idx, pre = 0, -1
    buffer = []
    for i, a in enumerate(arr[:, crit_axis]):
        if pre > a: tmp_idx += 1
        if tmp_idx == idx: buffer.append(i)
        elif tmp_idx > idx: break
        pre = a

    return arr[buffer].squeeze().tolist()


def get_pat_idx_from_unit(onoff, onoff_idx):
    buffer, pat_idx = 0, 0
    for oo, _, _ in onoff[:onoff_idx + 1]:
        if oo[0] < buffer:
            pat_idx += 1
        buffer = oo[0]
    return pat_idx
