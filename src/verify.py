import os
import sys
import random
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from matplotlib.collections import LineCollection


def plot_seg(data, label, label_plt=False, title='plot'):
    data = data.squeeze()
    label = label.squeeze()
    if data.shape != label.shape:        sys.exit('Data and label must be "one" sample')

    base = np.linspace(0, 2500, 2500)

    points = np.array([base, data]).T.reshape(-1, 1, 2)             # 2500, 1, 2
    segments = np.concatenate([points[:-1], points[1:]], axis=1)    # stack of 2 points along axis 1 (2500, 2, 2)

    cm = dict(zip(range(0, 8, 1), list(["black", "red", "green", "orange", 'crimson', '#DA70D6'])))
    colors = list(map(cm.get, label))

    lc = LineCollection(segments, colors=colors, linewidths=2)
    fig, ax = plt.subplots(figsize=(40, 10))
    if label_plt == True:
        ax.plot(label, '#D3D3D3')

    # legend
    a = mpatches.Patch(color='black', label='base_line')
    b = mpatches.Patch(color='red', label='P')  # #9ACD32
    c = mpatches.Patch(color='green', label='QRS')
    d = mpatches.Patch(color='orange', label='T')
    #     blue = mpatches.Patch(color='blue', label = 'R_peak')
    e = mpatches.Patch(color='crimson', label='vpc_QRS')
    f = mpatches.Patch(color='#DA70D6', label='vpc_T')
    #     red = mpatches.Patch(color='red', label='vpc_R_peak')

    plt.legend(handles=[a, b, c, d, e, f])

    ax.add_collection(lc)
    ax.autoscale()
    ax.margins(0.1)
    ax.set_xlabel('samples')
    ax.set_title(title)
    # plt.legend(['p', 'qrs','t','nw'], loc ='upper right')
    plt.tight_layout()
    plt.show()


def plot_compare(path, data, label=None, pred=None, rpeak=None, idx=9999, fn='Unknown', scaling='minmax', save=True):
    dd = data.squeeze()
    if isinstance(scaling, str):
        if scaling.lower() == 'dynamic':
            upper, lower = max(max(dd), 600), min(min(dd), -600)
        else:
            upper, lower = max(dd), min(dd)
        dd = (dd - lower) / (upper - lower)
    elif isinstance(scaling, list):
        dd = (dd - min(scaling)) / (max(scaling) - min(scaling))
    seg = (label, pred)
    palette = ['r', 'b']
    name = ['label', 'predict']
    legend = []

    plt.figure(100, figsize=(10, 4))
    plt.plot(dd * 2, 'k', linewidth=1)
    legend.append('data')
    for ii, (s, p) in enumerate(zip(seg, palette)):
        if s is None: continue
        s = s.squeeze()
        plt.plot(s/3-1, p, linewidth=0.5)
        legend.append(name[ii])

    if rpeak is not None and len(rpeak) > 0:
        for rp in rpeak.squeeze():
            plt.axvline(rp, linestyle=(0, (5, 3)), color='forestgreen', linewidth=0.2)

    plt.ylim([-1.2, 2.2])
    plt.legend(legend)
    plt.axhline(0, linestyle=(0, (5, 2)), color='grey', linewidth=0.5), plt.text(0, 0, 'VPC', fontsize=7)
    plt.axhline(-1 / 3, linestyle=(0, (5, 2)), color='grey', linewidth=0.5), plt.text(0, -1 / 3, 'APC', fontsize=7)
    plt.axhline(-2 / 3, linestyle=(0, (5, 2)), color='grey', linewidth=0.5), plt.text(0, -2 / 3, 'NSR', fontsize=7)
    plt.axhline(-3 / 3, linestyle=(0, (5, 2)), color='grey', linewidth=0.5), plt.text(0, -3 / 3, 'base', fontsize=7)
    plt.title('Idx: %d, Patient_ID: %s' % (idx, fn))
    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(path, '%s.png' % fn), dpi=80)
    else:
        plt.show()
    plt.close(100)


def set_seeds(seed):
    """ set random seeds """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

