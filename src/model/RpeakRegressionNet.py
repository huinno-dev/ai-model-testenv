import torch
import torch.nn as nn
import math


def get_model(model_cfg):
    # position arguments
    ch_in = model_cfg.ch_in
    width = model_cfg.width
    kernel_size = model_cfg.kernel_size

    # keyword arguments
    order = model_cfg.order if 'order' in model_cfg.__dir__() else 4
    depth = model_cfg.depth if 'depth' in model_cfg.__dir__() else 2
    stride = model_cfg.stride if 'stride' in model_cfg.__dir__() else 1
    regression_order = model_cfg.regression_order if 'regression_order' in model_cfg.__dir__() else 1
    embedding_dims = model_cfg.embedding_dims if 'embedding_dims' in model_cfg.__dir__() else 16
    input_length = model_cfg.resample_len if 'resample_len' in model_cfg.__dir__() else 64

    model = R_peak_regression_se_block(ch_in, width, kernel_size, depth=depth, order=order, stride=stride,
                                       regression_order=regression_order, embedding_dims=embedding_dims,
                                       input_length=input_length)
    return model


class R_peak_regression_se_block(nn.Module):
    def __init__(self,
                 ch_in: int,
                 width: int,
                 kernel_size: int or tuple,
                 depth: int = 4,
                 order: int = 1,
                 stride: int or tuple = 5,
                 regression_order : int = 4,
                 embedding_dims : int = 4096,
                 input_length: int or tuple = 200):
        super(R_peak_regression_se_block, self).__init__()
        self.ch_in = ch_in
        self.width = width
        self.kernel_size = kernel_size
        self.order = order
        self.depth = depth
        self.stride = stride
        self.emb_dim = embedding_dims
        self.reg_order = regression_order

        self.stem = down_layer_regression(ch_in, width, kernel_size, stride=stride, order=order)
        self.output_length = self.update_out_length(input_length)
        self.clipper = Clipped_ReLU(1)

        # Depth 별로 사용할 pooling, down, up list로 저장하기
        downs = []
        for d in range(1, depth):           # 1, 2, 3
            downs.append(down_layer_regression(width * d, width * (d + 1), kernel_size, stride=stride, order=order))
            self.output_length = self.update_out_length(self.output_length)
        self.downs = nn.ModuleList(downs)

        self.deep_embedding = self.embedding_block(self.output_length * width * depth, embedding_dims)
        self.regression_block = nn.Linear(self.output_length, 1)

    def update_out_length(self, length):
        return (length+(self.kernel_size-1)//2*2-(self.kernel_size-1)-1)//self.stride+1

    def embedding_block(self, in_length, out_length):
        self.output_length = out_length
        depth = self.reg_order
        block = []
        for i in range(depth):
            if i == 0:
                block.append(nn.Linear(in_length, int(out_length // 2 ** (depth - 1 - i))))
            else:
                block.append(nn.Linear(int(out_length // 2 ** (depth - i)), int(out_length // 2 ** (depth - 1 - i))))
            block.append(nn.ReLU())
        return nn.Sequential(*block)

    def forward(self, x):
        # -------------Encoder#------------- #
        for i_down in range(self.depth):        # 0, 1, 2, 3
            if i_down == 0:
                x = self.stem(x)
            else:
                x = self.downs[i_down - 1](x)

        flat = x.view(x.shape[0], -1)
        emb = self.deep_embedding(flat)
        return self.regression_block(emb)


def excitation_block(input_layer, number_of_excitation_channel):
    block = []
    block.append(nn.Linear(input_layer, number_of_excitation_channel))
    block.append(nn.ReLU())
    block.append(nn.Linear(number_of_excitation_channel, number_of_excitation_channel))
    block.append(nn.Sigmoid())
    return nn.Sequential(*block)


def down_layer_regression(ch_in, ch_out, kernel, stride, order):
    block = [CBR_block_regression(ch_in, ch_out, kernel, stride, 1)]
    for i in range(order):
        block.append(Res_block_regression(ch_out, ch_out, kernel, 1))
    return nn.Sequential(*block)


class Res_block_regression(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, dilation):
        super(Res_block_regression, self).__init__()

        self.cbr1 = CBR_block_regression(ch_in, ch_out, kernel_size, 1, dilation)
        self.cbr2 = CBR_block_regression(ch_out, ch_out, kernel_size, 1, dilation)
        self.se = SE_block(ch_out, ch_out, reduction=8)

    def forward(self, x):
        x_re = self.cbr1(x)
        x_re = self.cbr2(x_re)
        x_re = self.se(x_re)
        x_out = torch.add(x, x_re)
        return x_out


class CBR_block_regression(nn.Module):
    def __init__(self, ch_in, ch_out, kernel_size, stride, dilation):
        super(CBR_block_regression, self).__init__()
        pad = (kernel_size-1)//2

        self.conv1 = nn.Conv1d(ch_in, ch_out, kernel_size=kernel_size, stride=stride, dilation=dilation,
                               padding=pad, bias=True)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        out = self.relu(x)

        return out


class SE_block(nn.Module):
    def __init__(self, ch_in, ch_out, reduction):
        super(SE_block, self).__init__()
        self.pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Conv1d(ch_in, ch_out//reduction, kernel_size=(1,))
        self.fc2 = nn.Conv1d(ch_out//reduction, ch_out, kernel_size=(1,))
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        vec = self.pool(x)
        vec = self.relu(self.fc(vec))
        vec = self.sigmoid(self.fc2(vec))

        x = x * vec             # return x times vec instead of adding
        # x = torch.add(x, vec)
        return x


class Clipped_ReLU(nn.Module):
    def __init__(self, th):
        super(Clipped_ReLU, self).__init__()
        self.th = th
        self.act = nn.ReLU()

    def forward(self, x):
        return self.act(x) - self.act(x - self.th)
