from fastai.vision import *
from fastai.callbacks import SaveModelCallback
import os
from bengaliai.utils import Mish, to_Mish


class Loss_combine(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input, target):
        x1, x2, x3 = input
        y = target.long()
        return 2.0 * F.cross_entropy(x1, y[:, 0]) + F.cross_entropy(x2, y[:, 1]) + \
               F.cross_entropy(x3, y[:, 2])



