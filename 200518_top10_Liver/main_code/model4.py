import numpy as np
import module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, snp_len, base_node, drop_prob):
        super(Model, self).__init__()
        self.fc1 = nn.Sequential(nn.Linear(snp_len, base_node),
                                 nn.Dropout(drop_prob),
                                nn.ReLU())
        self.fc2 = nn.Linear(base_node, 1)

    def forward(self, data, target):
        h = self.fc1(data)
        h = self.fc2(h)
        l1 = torch.norm(self.fc1[0].weight, p=1) + torch.norm(self.fc2.weight, p=1)
        return h, l1
        