
from __future__ import absolute_import, print_function

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from encoding.nn import SyncBatchNorm

    _BATCH_NORM = SyncBatchNorm
   
except:
    # run below
    _BATCH_NORM = nn.BatchNorm2d

_BOTTLENECK_EXPANSION = 4


class _ConvBnReLU(nn.Sequential):
    """
    Cascade of 2D convolution, batch norm, and ReLU.
    """

    BATCH_NORM = _BATCH_NORM

    def __init__(self, in_ch, out_ch, kernel_size, stride, padding, dilation, relu=True):
        super(_ConvBnReLU, self).__init__()
        self.add_module("conv",nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, dilation, bias=False),)
        self.add_module("bn", _BATCH_NORM(out_ch, eps=1e-5, momentum=0.999))
        if relu:
            self.add_module("relu", nn.ReLU())


if __name__ == "__main__":
    model = ResNet(n_classes=1000, n_blocks=[3, 4, 23, 3])    
    model.eval()
    image = torch.randn(1, 3, 224, 224)

    print(model)
    print("input:", image.shape)
    print("output:", model(image).size())
