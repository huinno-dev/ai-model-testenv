import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from AI_common import nn_night as nnn


class BeatBase(nn.Module):
    def __init__(self):
        super(BeatBase, self).__init__()

    def gen_block(self, ch_in, ch_out, kernel_size, order, dilation=1, se_bias=False):
        """
        Return block, accumulated layers in-stage. Stack CBR 'order' times.
        :param ch_in:
        :param ch_out:
        :param kernel_size:
        :param order:
        :param dilation:
        :param se_bias:
        :return:
        """

        class Res(nn.Sequential):
            def __init__(self, *args):
                super(Res, self).__init__(*args)

            def forward(self, x):
                xx = x
                for module in self:
                    x = module(x)
                return x + xx

        blk = []
        for o in range(order):
            sequence = Res(self.unit_layer(ch_in, ch_out, kernel_size, dilation=dilation),
                           self.unit_layer(ch_out, ch_out, kernel_size, dilation=dilation),
                           SE(ch_out, reduction=8, bias=se_bias))
            blk.append(sequence)
        return nn.Sequential(*blk)

    def gen_head(self, c_in, kernel=None):
        if kernel is None: kernel = self.kernel_size
        return nn.Conv1d(c_in, self.ch_out, kernel_size=kernel, padding=(kernel - 1) // 2)

    def unit_layer(self, ch_in, ch_out, kernel_size, stride=1, dilation=1):
        """
        Return unit layer. Equal to CBR
        :param ch_in:
        :param ch_out:
        :param kernel_size:
        :param stride:
        :param dilation:
        :return:
        """
        conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=(stride,), dilation=(dilation,),
                          padding=dilation * (kernel_size - 1) // 2, bias=True)
        return nn.Sequential(*[conv1, nn.BatchNorm1d(ch_out), nn.ReLU()])


class BeatSegment(BeatBase):
    """
    Builder class for beat_segmentation which has U-Net-like structure.
    :param ch_in: int
        The channels-wise dimension of input tensor.
    :param ch_out: int
        The channels-wise dimension of output tensor, same as number of classes.
    :param width: int
        The channel expand factor.
    :param kernel_size: int
    :param depth: int
        The number of pooling layers. Must be 1 less than length of stride.
    :param order: int
        The number of blocks in stage. The stage means layer sequence in a depth.
    :param stride: list or tuple
        The scaling ratio for pooling layers.
    :param decoding: bool
        Optional.
        If True, build the decoding part of U-Net.
        If not, replace the decoding part of U-Net to up-sampling.
    :param expanding: bool, optional
        Optional.
        If True, input tensor padded 30 samples bi-side along the spatial axis to match the shape and product of stride.
    :param se_bias: bool, optional
        Optional.
        If True, SE modules have extra weights (bias).
    """
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 width: int,
                 kernel_size: int or tuple,
                 depth: int,
                 order: int,
                 stride: list or tuple,
                 decoding: bool,
                 **kwargs):
        super(BeatSegment, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.width = width
        self.kernel_size = kernel_size
        self.depth = depth
        self.order = order
        self.stride = stride
        self.decoding = decoding
        self.kwargs = kwargs

        self.expanding = kwargs['expanding'] if 'expanding' in kwargs.keys() else False
        self.se_bias = kwargs['se_bias'] if 'se_bias' in kwargs.keys() else False

        # Encoder (down-stream)
        self.pools = nn.ModuleList()
        self.enc = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                self.enc.append(nn.Sequential(self.unit_layer(ch_in, width, kernel_size),
                                              self.gen_block(width, width, kernel_size, order, se_bias=self.se_bias)))
            else:
                if d == 1:
                    c_in, c_out, s = width * d, width * (d + 1), stride[d - 1]
                else:
                    c_in, c_out, s = ch_in + width * d, width * (d + 1), stride[d - 1]
                self.pools.append(nn.AvgPool1d(np.prod(stride[:d]), np.prod(stride[:d])))
                self.enc.append(nn.Sequential(self.unit_layer(c_in, c_out, kernel_size, stride=s),
                                              self.gen_block(c_out, c_out, kernel_size, order, se_bias=self.se_bias)))

        self.un_pools = nn.ModuleList()
        for s in stride[::-1]:
            self.un_pools.append(nn.Upsample(scale_factor=s, mode='linear', align_corners=True))

        # Decoder (up-stream)
        if self.decoding:
            self.dec = nn.ModuleList()
            for d in reversed(range(1, depth)):
                c_in, c_out = (2*d+1) * width, d * width
                self.dec.append(nn.Sequential(self.unit_layer(c_in, c_out, kernel_size),
                                              self.gen_block(c_out, c_out, kernel_size, order, se_bias=self.se_bias)))

        self.head = self.gen_head(width if decoding else width * depth, kernel=kernel_size)

    def forward(self, x):
        if self.expanding:
            x = F.pad(x, (30, 30), mode='constant', value=0)

        keep = x if x.shape[1] == 1 else x[:, x.shape[1] // 2, :].unsqueeze(1)
        for_skip, for_feat = [], []
        # -------------Encoder#------------- #
        for d in range(self.depth):        # 0, 1, 2, 3
            if d == 0:
                for_skip.append(self.enc[d](x))
            elif d == 1:
                for_skip.append(self.enc[d](for_skip[-1]))
            else:
                for_skip.append(self.enc[d](torch.cat([for_skip[-1], self.pools[d - 2](keep)], 1)))

        # -------------Decoder#------------- #
        if not self.decoding:
            out = self.head(for_skip[-1])
            for un_pool in self.un_pools:
                out = un_pool(out)
        else:
            for_feat.append(for_skip.pop(-1))
            for_skip.reverse()
            for d in range(self.depth - 1):     # 0, 1, 2
                concat = torch.cat((for_skip[d], self.un_pools[d](for_feat[d])), dim=1)
                for_feat.append(self.dec[d](concat))
            out = self.head(for_feat[-1])
        if self.expanding:
            out = tuple([o[:, :, 30:-30] for o in out]) if isinstance(out, tuple) else out[:, :, 30:-30]
        return out


class BeatSegment2(BeatBase):
    def __init__(self,
                 ch_in: int,
                 ch_out: int,
                 width: int,
                 kernel_size: int or tuple,
                 depth: int,
                 order: int,
                 stride: list or tuple,
                 decoding: bool,
                 **kwargs):
        super(BeatSegment2, self).__init__()

        self.ch_in = ch_in
        self.ch_out = ch_out
        self.width = width
        self.kernel_size = kernel_size
        self.depth = depth
        self.order = order
        self.stride = stride
        self.decoding = decoding
        self.kwargs = kwargs

        self.expanding = kwargs['expanding'] if 'expanding' in kwargs.keys() else False
        self.se_bias = kwargs['se_bias'] if 'se_bias' in kwargs.keys() else False

        # Encoder (down-stream)
        self.pools = nn.ModuleList()
        self.enc = nn.ModuleList()
        for d in range(depth):
            if d == 0:
                self.enc.append(nn.Sequential(self.unit_layer(ch_in, width, kernel_size),
                                              self.gen_block(width, width, kernel_size, order, se_bias=self.se_bias)))
            else:
                if d == 1:
                    c_in, c_out, s = width * d, width * (d + 1), stride[d - 1]
                else:
                    c_in, c_out, s = ch_in + width * d, width * (d + 1), stride[d - 1]
                self.pools.append(nn.AvgPool1d(np.prod(stride[:d]), np.prod(stride[:d])))
                self.enc.append(nn.Sequential(self.unit_layer(c_in, c_out, kernel_size, stride=s),
                                              self.gen_block(c_out, c_out, kernel_size, order, se_bias=self.se_bias)))

        self.un_pools = nn.ModuleList()
        for s in stride[::-1]:
            self.un_pools.append(nn.Upsample(scale_factor=s, mode='linear', align_corners=True))

        # Decoder (up-stream)
        if self.decoding:
            self.dec = nn.ModuleList()
            for d in reversed(range(1, depth)):
                c_in, c_out = (2*d+1) * width, d * width
                self.dec.append(nn.Sequential(self.unit_layer(c_in, c_out, kernel_size),
                                              self.gen_block(c_out, c_out, kernel_size, order, se_bias=self.se_bias)))

        self.head = self.gen_head(width if decoding else width * depth, kernel=kernel_size)

    def unit_layer(self, ch_in, ch_out, kernel_size, stride=1, dilation=1):
        conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=(stride,), dilation=(dilation,),
                          padding=dilation * (kernel_size - 1) // 2, bias=True)
        return nn.Sequential(*[conv1, nn.BatchNorm1d(ch_out), nn.ReLU()])

    def forward(self, x):
        if self.expanding:
            x = F.pad(x, (30, 30), mode='constant', value=0)

        for_skip, for_feat = [], []

        # -------------Encoder#------------- #
        for d in range(self.depth):        # 0, 1, 2, 3
            if d == 0:
                for_skip.append(self.enc[d](x))
            elif d == 1:
                for_skip.append(self.enc[d](for_skip[-1]))
            else:
                for_skip.append(self.enc[d](torch.cat([for_skip[-1], self.pools[d - 2](x)], 1)))

        # -------------Decoder#------------- #
        if not self.decoding:
            out = self.head(for_skip[-1])
            for un_pool in self.un_pools:
                out = un_pool(out)
        else:
            for_feat.append(for_skip.pop(-1))
            for_skip.reverse()
            for d in range(self.depth - 1):     # 0, 1, 2
                concat = torch.cat((for_skip[d], self.un_pools[d](for_feat[d])), dim=1)
                for_feat.append(self.dec[d](concat))
            out = self.head(for_feat[-1])
        if self.expanding:
            out = tuple([o[:, :, 30:-30] for o in out]) if isinstance(out, tuple) else out[:, :, 30:-30]
        return out


class RPeakRegress(BeatBase):
    def __init__(self,
                 dim_in: int,
                 ch_in: int,
                 width: int,
                 kernel_size: int or tuple,
                 order: int,
                 depth: int,
                 stride: int,
                 head_depth: int,
                 embedding_dims: int,
                 **kwargs):
        super(RPeakRegress, self).__init__()

        self.width = width
        self.kernel_size = kernel_size
        self.order = order
        self.depth = depth
        self.stride = stride
        self.head_depth = head_depth
        self.dim = dim_in
        self.embedding_dims = embedding_dims
        self.kwargs = kwargs

        self.se_bias = kwargs['se_bias'] if 'se_bias' in kwargs.keys() else False

        # Encoder (down-stream)
        enc = []
        for d in range(depth):
            c_in = ch_in if d == 0 else width * d
            enc.append(self.unit_layer(c_in, width * (d + 1), self.kernel_size, self.stride))
            enc.append(self.gen_block(width * (d + 1), width * (d + 1), kernel_size, order=order, se_bias=self.se_bias))
        self.enc = nn.Sequential(*enc)

        self.update_dimension()
        self.emb_layer = self.gen_embedding()
        self.head = nn.Linear(self.embedding_dims, 1)
        self.clipper = nnn.ClippedReLU(1)

    def update_dimension(self):
        for _ in range(self.depth):
            self.dim = (self.dim+(self.kernel_size-1)//2*2-(self.kernel_size-1)-1)//self.stride+1
            # self.dim = 1 + (self.dim + 2*(self.kernel_size//2) - self.kernel_size) // self.stride

    def gen_embedding(self):
        block = []
        for h in range(self.head_depth):
            d_in = self.dim * self.width * self.depth if h == 0 else int(self.dim // 2 ** (self.head_depth - h))
            block.append(nn.Linear(d_in, self.embedding_dims))
            block.append(nn.ReLU())
        return nn.Sequential(*block)

    def unit_layer(self, ch_in, ch_out, kernel_size, stride=1, dilation=1):
        conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=(stride,), dilation=(dilation,),
                          padding=dilation * (kernel_size - 1) // 2, bias=True)
        return nn.Sequential(*[conv1, nn.ReLU()])

    def forward(self, x):
        # -------------Encoder#------------- #
        x = self.enc(x)
        flat = x.view(x.shape[0], -1)
        emb = self.emb_layer(flat)
        return self.head(emb)


class SE(nn.Module):
    def __init__(self, ch: int, reduction: int = 8, spatial: bool = False, bias: bool = False):
        super(SE, self).__init__()
        self.reduction = reduction
        self.spatial = spatial
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        if spatial:
            self.ch_pool = nn.Conv1d(ch, 1, (1,))
        else:
            assert ch % reduction == 0, f'Received invalid arguments. The "reduction" must be a divisor of "B".'
            self.pool = nn.AdaptiveAvgPool1d(1)
            self.fc = nn.Conv1d(ch, ch // reduction, kernel_size=(1,), bias=bias)
            self.fc2 = nn.Conv1d(ch // reduction, ch, kernel_size=(1,), bias=bias)

    def channel_wise(self, x):
        vec = self.relu(self.fc(self.pool(x)))
        vec = self.sigmoid(self.fc2(vec))
        return x * vec

    def spatial_wise(self, x):
        attn_map = self.sigmoid(self.ch_pool(x))
        return x * attn_map

    def forward(self, x):
        if self.spatial:
            return self.spatial_wise(x)
        else:
            return self.channel_wise(x)
