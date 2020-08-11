import torch.nn.functional as F
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self, args):
        super(Loss, self).__init__()
        self.args = args