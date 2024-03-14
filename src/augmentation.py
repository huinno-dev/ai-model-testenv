import cv2
import numpy as np
from biosppy.signals import tools as st
from scipy.stats import skewnorm
import scipy.signal as sig
import torch
import random


# Noise Augmentation
def _x_watts(data):
    x_watts = data ** 2
    return x_watts


def make_SNR(x_volts, target_snr_db=5, mode=2):
    # Adding noise using target SNR
    # Set a target SNR
    # Calculate signal power and convert to dB
    # mode 1 is "recomended for spike type noises"
    if mode == 2:  # Squared
        x_watts = _x_watts(x_volts)
        sig_avg_watts = np.mean(x_watts)
    elif mode == 1:  # absolute value (for spike noise type)
        x_watts = np.abs(x_volts)
        sig_avg_watts = np.max(x_watts)
    if sig_avg_watts < 0.000001: # Zero signal이 들어오면 그냥 1 mV 기준의 noise 생성
        sig_avg_db = 0
    else:    
        sig_avg_db = 10 * np.log10(sig_avg_watts + 1e-8)
    # Calculate noise according to [2] then convert to watts
    noise_avg_db = sig_avg_db - target_snr_db  # SNR

    return noise_avg_db


def make_noise_power(x_volts, target_snr_db=5, mode=2):  # target_snr_db가 작으면 더 지직거림
    # mode 1 is "recomended for spike type noises"
    noise_avg_db = make_SNR(x_volts, target_snr_db=target_snr_db, mode=mode)
    noise_avg_watts = 10 ** (noise_avg_db / 10)  # Pn : noise_power

    return noise_avg_watts


def reverse_signal(x_volts, target_snr_db=5, rand_amp_dB=False, SEED=None):
    rev_signal = x_volts.copy()
    rev_signal *= -1
    return rev_signal


def select_noise_segment(len_x, min_interval=None, SEED=None):
    if min_interval is None:
        interval = np.random.choice(int(len_x * 0.95))
        interval += int(len_x * 0.05)
    else:
        interval = np.random.choice(int(len_x) - min_interval)
        interval += int(min_interval)
    interval_start = np.random.choice(len_x - interval)
    return interval_start, interval


def additive_gaussian_noise(x_volts, target_snr_db=5, rand_amp_dB=False, interval_mode=0, interval_start=None,
                            interval=None, SEED=None):
    """
    Usage example: awgn_volts = AWGN_augmentation_random(x_volts, target_snr_db = 20)
    """
    mean_noise = 0
    # coeff = np.random.uniform(10, 20)

    x_volts = x_volts.squeeze()
    x_len = len(x_volts)
    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    C = make_noise_power(filtered_volts, target_snr_db=target_snr_db)
    if SEED:
        np.random.seed(SEED)

    if interval_mode == 0:
        interval_type = np.random.choice([0, 1], p=[0.66, 0.34])  # 0 : random partial, 1 : entire segment
        if interval_type:
            interval = x_len
            interval_start = 0
        else:
            interval_start, interval = select_noise_segment(x_len, SEED=SEED)
    gaussian_noise = np.random.normal(mean_noise, np.sqrt(C), interval)  # C corresponds to variance. The second

    noise = [0] * x_len
    for i in range(interval_start, interval_start + interval):
        noise[i] = gaussian_noise[i - interval_start]
    # awgn = np.array(x_volts) + np.array(noise)

    return np.array(noise)


def additive_random_spike(x_volts, target_snr_db=3, rand_amp_dB=False, SEED=None):
    """
    RS - Random Spikes
    Usage: rs_volts = RS_augmentation(x_volts, f = 20, C = 50)
    C가 클수록 더 높게 spike, f가 커질수록 자주 spike
    """
    N = np.random.randint(1, 10)
    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    C = make_noise_power(filtered_volts, target_snr_db=target_snr_db, mode=1)

    Sp = np.array([0, 0.15, 1.5, -0.25, 0.15]) + 2
    Sp = Sp / np.max(Sp)
    if SEED:
        np.random.seed(SEED)

    x_len = x_volts.shape[-1]
    f_list = np.random.choice(np.array(range(2, x_len - 2)), N)
    RS_noise = [0] * len(x_volts)
    for i in f_list:
        for index, j in enumerate([-2, -1, 0, 1, 2]):
            noise = Sp[index]
            RS_noise[i + j] = C * noise
    # RS_noise = np.array(RS_noise)
    # rs_volts = x_volts + RS_noise

    return np.array(RS_noise)


def PN_augmentation(x_volts, target_snr_db=5, rand_amp_dB=False, fs=250.0, f=60, interval_mode=0, interval_start=None,
                    interval=None):
    """
    PN - Power line Noise
    pn_volts, PN_noise = PN_augmentation(x_volts, fs = 500, f = 60, use_Pn = True)
    """
    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    C = make_noise_power(filtered_volts, target_snr_db=target_snr_db)
    PN_noise = []
    for n in range(len(x_volts)):
        noise = np.sqrt(2 * C) * np.cos(2 * np.pi * f / fs * n)
        PN_noise.append(noise)
    # PN_noise = np.array(PN_noise)
    # pn_volts = x_volts + PN_noise

    return np.array(PN_noise)


def additive_baseline_wandering(x_volts, target_snr_db=-5, rand_amp_dB=False, fs=250.0, interval_mode=0,
                                interval_start=None, interval=None, SEED=None):
    """
    BW - Baseline Wander
    bw_volts, BW_noise = BW_augmentation(x_volts, fs = 500, f = 0.2, C = 10)
    C가 크면 진폭이 커짐, f가 커지면 더 자주 휨
    """
    if SEED:
        np.random.seed(SEED)
    theta = np.random.uniform(0, np.pi * 2)
    f = np.random.uniform(0.01, 0.2)

    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    C = make_noise_power(filtered_volts, target_snr_db=target_snr_db)
    BW_noise = []
    for n in range(len(x_volts)):
        noise = np.sqrt(C) * np.cos(2 * np.pi * f / fs * n - theta)
        BW_noise.append(noise)
    # BW_noise = np.array(BW_noise)
    # bw_volts = x_volts + BW_noise

    return np.array(BW_noise)


def additive_natural_baseline_wandering(x_volts, target_snr_db=-3, rand_amp_dB=False, fs=250.0, interval_mode=0,
                                        interval_start=None, interval=None, SEED=None):
    x_volts = x_volts.squeeze().copy()
    x_len = x_volts.shape[0]

    if SEED:
        np.random.seed(SEED)

    if interval_mode == 0:
        interval_start, interval = select_noise_segment(x_len, SEED=SEED)

    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    C = make_noise_power(filtered_volts, target_snr_db=target_snr_db)

    noise_array = []
    temp = 0
    window_function = sig.windows.tukey(interval)
    for _ in range(interval):
        temp = np.random.randn() + temp
        noise_array.append(temp)

    noise_array = sig.filtfilt(*sig.butter(2, [0.004, 0.015], btype='bandpass'), noise_array)
    noise_array /= np.std(noise_array)

    bnw_volts = np.zeros(x_len)
    bnw_volts[interval_start:interval_start + interval] += np.sqrt(C) * noise_array * window_function

    return bnw_volts


def attenuated_spike_noise(x_volts, target_snr_db=8, rand_amp_dB=False, fs=250.0, lambda1=0.6, lambda2=0.8,
                           rand_polarity=True, interval_mode=0, interval_start=None, interval=None, SEED=None):
    x_shape = x_volts.shape
    x_volts = x_volts.squeeze()
    x_len = x_volts.shape[0]

    if SEED:
        np.random.seed(SEED)

    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    C = make_noise_power(filtered_volts, target_snr_db=target_snr_db, mode=1)
    spike_period = np.random.randint(2, 7)  # number of spike / sec

    if interval_mode == 0:
        interval_start, interval = select_noise_segment(x_len, SEED=SEED)

    noise_signal = np.zeros(interval)
    noise_signal[int(fs // spike_period) // 2:-int(fs // spike_period):int(fs // spike_period)] = 1

    for i in range(interval):
        if i == 0:
            continue
        else:
            if noise_signal[i] >= noise_signal[i - 1]:
                noise_signal[i] = lambda1 * noise_signal[i - 1] + (1 - lambda1) * noise_signal[i]
            else:
                noise_signal[i] = lambda2 * noise_signal[i - 1] + (1 - lambda2) * noise_signal[i]

    noise_signal /= (np.max(noise_signal) + 1e-6)

    asp_volts = np.zeros(x_len)
    if rand_polarity:
        asp_volts[interval_start:interval_start + interval] += ((-1) ** np.random.randint(2)) * C * noise_signal
    else:
        asp_volts[interval_start:interval_start + interval] += C * noise_signal

    return asp_volts.reshape(x_shape)


def additive_baseline_like_vpc(x_volts, target_snr_db=3, rand_amp_dB=False, fs=250.0, SEED=None):
    x_shape = x_volts.shape
    x_len = x_volts.shape[0]
    if SEED:
        np.random.seed(SEED)
    interval = np.random.randint(fs * 0.2, fs)
    interval_start = np.random.choice(len(x_volts) - interval)
    a = np.random.randint(-20, -2)
    rv = skewnorm(a)
    x = np.linspace(skewnorm.ppf(0.01, a), skewnorm.ppf(0.999, a), interval)
    vpc_bw_degree = rv.pdf(x) ** 2
    vpc_bw_degree = vpc_bw_degree / np.linalg.norm(vpc_bw_degree)

    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    # C = make_noise_power(filtered_volts, target_snr_db=target_snr_db)
    C = 1.0
    vpc_bw_noise = np.sqrt(C) * vpc_bw_degree

    vpc_volts = np.zeros(x_len)
    vpc_volts[interval_start:interval_start + interval] += vpc_bw_noise

    return vpc_volts.reshape(x_shape)


def additive_emg(x_volts, total_emg_list, target_snr_db=8, rand_amp_dB=False, fs=250.0, scale_factor=0.5, SEED=None):
    x_len = x_volts.shape[-1]

    if SEED:
        np.random.seed(SEED)

    total_emg1, total_emg2 = total_emg_list[0].copy(), total_emg_list[1].copy()
    if np.random.rand() > 0.5:
        total_emg = total_emg1
    else:
        total_emg = total_emg2

    emg_noise = np.zeros(x_len)

    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    rand_interval_num = np.random.randint(5, 10)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    C = make_noise_power(filtered_volts, target_snr_db=target_snr_db, mode=1)

    for num in range(rand_interval_num):
        # 추출할 전체 emg에서 interval 추출
        interval = np.random.randint(fs * 0.5, fs * 1)
        # 전체 emg에서 구간 추출을 위한 시작점
        emg_rand_starting_pt = np.random.randint(0, len(total_emg) - interval)
        # emg 구간 추출
        interval_emg = total_emg[emg_rand_starting_pt:emg_rand_starting_pt + interval]
        # Normalizing by standard deviation
        if np.std(interval_emg) == 0:
            continue
        scaled_noise = interval_emg / np.std(interval_emg)
        scaled_noise *= C
        # 입력 ecg 신호에서 emg를 집어넣을 시작점
        input_rand_starting_pt = np.random.randint(0, len(x_volts) - interval)
        emg_noise[input_rand_starting_pt:input_rand_starting_pt + interval] += scaled_noise * sig.windows.hann(interval)

    return emg_noise


def additive_baseline_like_vpc2(x_volts, target_snr_db=-3, rand_amp_dB=False, fs=250.0, SEED=None):
    x_shape = x_volts.shape
    x_len = x_volts.shape[0]
    if SEED:
        np.random.seed(SEED)
    interval = np.random.randint(fs * 0.2, fs)
    interval_start = np.random.choice(len(x_volts) - interval)
    
    spikes = np.zeros(interval)
    bandwidth_control = np.random.gamma(3)
    negative_peak_offset = 2
    first_peak = np.random.chisquare(1)
    second_peak = -6
    spikes[2] = first_peak
    spikes[2+negative_peak_offset] = second_peak
    vpc_bw_noise = sig.lfilter(*sig.butter(2, 0.01*bandwidth_control, btype='lowpass'), spikes)

    vpc_bw_degree = vpc_bw_noise/np.max(np.abs(vpc_bw_noise))
    

    filtered_volts = sig.filtfilt(*sig.butter(2, [0.004, 0.4], btype='bandpass'), x_volts)
    if rand_amp_dB:
        target_snr_db += np.random.randn() * rand_amp_dB
    C = make_noise_power(filtered_volts, target_snr_db=target_snr_db, mode=1)
    vpc_bw_degree = np.sqrt(C) * vpc_bw_degree

    vpc_volts = np.zeros(x_len)
    vpc_volts[interval_start:interval_start + interval] += vpc_bw_degree
    
    return vpc_volts.reshape(x_shape)


def noise_augmentation(x: torch.Tensor, emg_signals: list, augmentation_list=None,
                       rand_amp_dB = 0.0, pmf_for_noise_type=None, noise_severity_dB=0.0):
    """
    Randomly apply augmentation methods
    Args:
        mini_batch (torch.tensor or numpy.array): minibatch, size might be (minibatch_size) X (1) X (2500)
        augmentation_list (list, optional): A list of augmentation function to use. Defaults to [ additive_gaussian_noise, additive_random_spike, PN_augmentation, additive_baseline_wandering, additive_natural_baseline_wandering, additive_baseline_like_vpc, attenuated_spike_noise ].
        rand_amp_dB (float or list, optional): Random perturbation of target SNR. To control each augmentation methods, use list of float, such as, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0]. Defaults to 0.0.
        pmf_for_noise_type (list, optional): Probabilities each augmentation fuction to be selected. Defaults to [0.2, 0.125, 0.125, 0.15, 0.25, 0.1, 0.05].
        noise_severity_dB (float or list, optional): Not implemented. Parameter to control noise severity. To control each augmentation methods, use list of float, such as, [1.0, 0.0, 2.0, 0.0, 0.0, 0.0]. Defaults to 0.0.
    """

    if augmentation_list is None:
        augmentation_list = [additive_gaussian_noise, additive_random_spike, PN_augmentation,
                             additive_natural_baseline_wandering, additive_baseline_like_vpc2, attenuated_spike_noise,
                             additive_emg]

    if pmf_for_noise_type is None:
        pmf_for_noise_type = (1 - sum(list(map(lambda n: 0.5**n, range(1, 9)))),
                              1 / 2, 1 / 2 ** 2, 1 / 2 ** 3, 1 / 2 ** 4, 1 / 2 ** 5, 1 / 2 ** 6,
                              1 / 2 ** 7, 1 / 2 ** 8)

    if len(pmf_for_noise_type) != len(augmentation_list)+1:
        remain = [1 - sum(list(map(lambda n: 0.5 ** n, range(1, len(augmentation_list)+1))))]
        coef = [0.5**(i+1) for i in range(len(augmentation_list))]
        pmf_for_noise_type = tuple(remain + coef)

    # aug_method_choice =0

    # 몇개의 함수를 중첩할 것인지 choice : 0~len
    aug_n = np.random.choice(range(len(augmentation_list) + 1), p=pmf_for_noise_type)

    if aug_n == 0:  # 함수 적용 없음
        return x
    else:  # 중첩 적용할 함수들 choice
        np_x = x.numpy().squeeze()
        augmentation_noise_array = []
        aug_method_choice = np.random.choice(len(augmentation_list), aug_n, replace=False)

        # 함수 중첩
        for i in aug_method_choice:
            if i == len(augmentation_list) - 1:
                augmentation_noise_array.append(
                    augmentation_list[i](np_x, total_emg_list=emg_signals))
            else:
                augmentation_noise_array.append(
                    augmentation_list[i](np_x, rand_amp_dB=rand_amp_dB))
        np_x += np.sum(augmentation_noise_array, axis=0)/aug_n
        return torch.tensor(np_x, dtype=torch.float32).unsqueeze(0)


def pseudo_tachy(signal, label, rpeaks):
    if len(rpeaks) == 0 or len(rpeaks) > 15 * signal.shape[-1] / 2500:        # pseudo_tachy는 HR 180 미만 data만 생성
        return signal, label
    signal_, label_ = signal.squeeze(), label.squeeze()
    rp_first, rp_last = rpeaks[0], rpeaks[-1]
    # In case of index error, return original signal
    try:
        if (label_[rp_first] != label_[rp_last]) or max(label_) == 3:
            return signal, label_.reshape(1, -1)
    except:
        return signal, label

    head_signal, head_label = signal_[:rp_last], label_[:rp_last]
    tail_signal, tail_label = signal_[rp_first:], label_[rp_first:]

    long_signal = np.concatenate((head_signal, tail_signal))
    long_label = np.concatenate((head_label, tail_label))
    # offset = rp_last - rp_first
    # long_rpeaks = np.concatenate((rpeaks, rpeaks[1:] + offset))

    # merged_signal = -sig.resample(long_signal, len(signal_))
    ratio = len(signal_) / len(long_signal)
    merged_signal = cv2.resize(long_signal.reshape(1, -1), dsize=(0, 0), fx=ratio, fy=1, interpolation=cv2.INTER_LINEAR)
    merged_signal = torch.tensor(merged_signal, dtype=torch.float32)
    merged_label = sig.resample(long_label, len(label_))
    merged_label = torch.tensor(abs(merged_label.round()), dtype=torch.long).reshape(1, -1)
    # cutting edge
    for i_l in range(0, len(merged_label) - 1, 2):
        if merged_label[i_l]:
            if merged_label[i_l] > merged_label[i_l+1]:
                merged_label[i_l+1] = merged_label[i_l]
    for i_l in range(len(merged_label) - 1, 0, -2):
        if merged_label[i_l]:
            if merged_label[i_l] > merged_label[i_l - 1]:
                merged_label[i_l - 1] = merged_label[i_l]

    # merged_rpeaks = long_rpeaks / (len(signal_) / len(long_signal))

    return merged_signal, merged_label


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    tmp = np.linspace(0, 2 * np.pi, 500)
    tmp = np.cos(tmp).reshape(1, -1)
    stretch = cv2.resize(tmp, dsize=(0, 0), fx=1.2, fy=1, interpolation=cv2.INTER_LINEAR)
    squeeze = cv2.resize(tmp, dsize=(0, 0), fx=0.8, fy=1, interpolation=cv2.INTER_LINEAR)

    plt.plot(stretch.squeeze())
    plt.plot(squeeze.squeeze())
    plt.show()


