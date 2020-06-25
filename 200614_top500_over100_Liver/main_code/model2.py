import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, k_len, input_ch_num):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(input_ch_num, input_ch_num, kernel_size=(3, k_len), stride=1, groups=input_ch_num),
            nn.LeakyReLU(negative_slope=0.1))
        self.fc1 = nn.Linear(input_ch_num, 1)

    def forward(self, data, target):
        x = self.conv1(data)
        x = x.view(x.size()[0], x.size()[1])
        h = self.fc1(x)
        l1 = torch.norm(self.fc1.weight, p=1)
        return h, l1
