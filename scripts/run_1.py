import torch
from torch import nn
from msconv2d import MSConv2d
from tile_concat import tile_and_concatenate


def run_msconv_1():
    nchw = torch.rand(10, 3, 7, 8)  # 7x8 image with 3 (RGB) channels, 10 images per batch
    nch = torch.rand(10, 4, 7)      # 4 row-wise metadata dimensions per image
    ncw = torch.rand(10, 5, 8)      # 5 column-wise metadata dimensions per image
    nc = torch.rand(10, 6)          # 6 scalar metadata entries per image
    tac_result = tile_and_concatenate(nchw, nch, ncw, nc)
    print("Tensor shape after tile-and-concatenate:", tac_result.shape)

    # 11 output channels
    conv = nn.Conv2d(3+4+5+6, 11, kernel_size=3)
    conv_result = conv(tac_result)
    print("Tensor shape after tile-and-concatenate and Conv2d:", conv_result.shape)

    msconv = MSConv2d([3, 4, 5, 6], 11, kernel_size=3)
    msconv_result = msconv(nchw, nch, ncw, nc)
    print("Tensor shape after MSConv2d:", msconv_result.shape)


run_msconv_1()