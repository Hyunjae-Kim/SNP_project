import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, snp_len):
        super(Model, self).__init__()
        self.fc1 = nn.Linear(snp_len, 1)

    def forward(self, data, target):
        h = self.fc1(data)
        l1 = torch.norm(self.fc1.weight, p=1)
        return h, l1
        