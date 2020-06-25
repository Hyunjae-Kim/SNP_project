import numpy as np
import module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, snp_len, base_node):
        super(Model, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(snp_len, base_node*2),
                                nn.BatchNorm1d(base_node*2),
                                nn.LeakyReLU(negative_slope=0.2))
        self.fc2 = nn.Sequential(nn.Linear(base_node*2, base_node),
                                 nn.BatchNorm1d(base_node),
                                nn.LeakyReLU(negative_slope=0.2))
        self.fc3 = nn.Linear(base_node, 1)

    def forward(self, data, target):
        h = self.fc1(data)
        h = self.fc2(h)
        h = self.fc3(h)
        return h
        