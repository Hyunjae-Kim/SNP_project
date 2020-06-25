import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, snp_len, base_node):
        super(Model, self).__init__()
        self.base_node = base_node
        self.fc1 = nn.Sequential(nn.Linear(snp_len, base_node*4),
                                 nn.Dropout(0.5),
                                nn.LeakyReLU())
        self.fc2 = nn.Sequential(nn.Linear(base_node*4, base_node*2),
                                 nn.Dropout(0.5),
                                nn.LeakyReLU())
        self.fc3 = nn.Sequential(nn.Linear(base_node*2, base_node),
                                 nn.Dropout(0.5),
                                nn.LeakyReLU())
        self.fc4 = nn.Linear(base_node, 1)

    def forward(self, data, target):
        h = self.fc1(data)
        h = self.fc2(h)
        h = self.fc3(h)
        h = self.fc4(h)
        l1 = (torch.norm(self.fc1[0].weight, p=1) + torch.norm(self.fc2[0].weight, p=1) + \
                torch.norm(self.fc3[0].weight, p=1) + torch.norm(self.fc4.weight, p=1))/self.base_node
        return h, l1