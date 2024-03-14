import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair


class ChannelPool(nn.Module):
    def __init__(self, cardinality):
        super(ChannelPool, self).__init__()
        self.cardinality = cardinality

    def forward(self, in_tensor):
        stacks = []
        ch_in = in_tensor.shape[1]
        assert ch_in % self.cardinality == 0
        sub_size = ch_in // self.cardinality
        for i in range(self.cardinality):
            sub = in_tensor[:, i*sub_size:(i+1)*sub_size]
            stacks.append(torch.mean(sub, dim=1, keepdim=True))
        return torch.cat(stacks, dim=1)


class MedianPool1d(nn.Module):
    """ Median pool (usable as median filter when stride=1) module.

    Args:
         kernel_size: size of pooling kernel, int or 2-tuple
         stride: pool stride, int or 2-tuple
         padding: pool padding, int or 4-tuple (l, r, t, b) as in pytorch F.pad
         same: override padding and enforce same padding, boolean
    """

    def __init__(self, kernel_size=3, stride=1, padding=0, same=False):
        super(MedianPool1d, self).__init__()
        self.k = kernel_size
        self.stride = stride
        self.padding = _pair(padding)  # convert to l, r
        self.same = same

    def _padding(self, x):
        if self.same:
            iw = x.size()[2:]
            if iw % self.stride == 0:
                pw = max(self.k - self.stride, 0)
            else:
                pw = max(self.k - (iw % self.stride), 0)
            pl = pw // 2
            pr = pw - pl
            padding = (pl, pr)
        else:
            padding = self.padding
        return padding

    def forward(self, x):
        # using existing pytorch functions and tensor ops so that we get autograd,
        # would likely be more efficient to implement from scratch at C/Cuda level
        x = F.pad(x, self._padding(x), mode='reflect')
        x = x.unfold(2, self.k, self.stride)
        x = x.contiguous().view(x.size()[:3] + (-1,)).median(dim=-1)[0]
        return x


class Post_layer(nn.Module):
    def __init__(self, ops: str, kernel_size: list):
        super(Post_layer, self).__init__()
        if len(ops) != len(kernel_size):
            print("The lengths of the arguments must match.")
            return
        morph_list = []
        for op, ker in zip(ops, kernel_size):
            if op.lower() == 'c':
                morph_list += [MorphLayer(ker, 'dilation'), MorphLayer(ker, 'erosion')]
            elif op.lower() == 'o':
                morph_list += [MorphLayer(ker, 'erosion'), MorphLayer(ker, 'dilation')]
            else:
                print("Unexpected operation keyword.")
        self.post = nn.Sequential(*morph_list)
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x):
        temp = []
        for i in range(x.shape[1]):
            temp.append(self.post(x[:, i, :].unsqueeze(1)))

        return torch.concat(temp, dim=1)


class MorphLayer(nn.Module):
    def __init__(self, kernel_size, morph_type):
        super(MorphLayer, self).__init__()

        self.morph_type = morph_type

        self.conv = nn.Conv1d(1, kernel_size, kernel_size, bias=False, padding=(kernel_size - 1) // 2,
                              padding_mode='replicate')
        kernel = torch.zeros((kernel_size, 1, kernel_size), dtype=torch.float)
        for i in range(kernel_size):
            kernel[i][0][i] = 1
        self.conv.weight.data = kernel

    def forward(self, x):
        x = self.conv(x)
        if self.morph_type == 'erosion':
            return torch.min(x, 1)[0].unsqueeze(1)
        elif self.morph_type == 'dilation':
            return torch.max(x, 1)[0].unsqueeze(1)


