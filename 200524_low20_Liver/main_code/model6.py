import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, snp_len, n_channel):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_channel, kernel_size=(3,32), stride=(1,16)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(n_channel))
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_channel, n_channel, kernel_size=8, stride=4),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(n_channel))
        self.fc1 = nn.Linear(n_channel*(int((int(snp_len/16)-1)/4)-1), 1)

    def forward(self, data, target):
        x = self.conv1(data)
        x = x.view(x.size()[0], x.size()[1],-1)
        x = self.conv2(x)
        x_ = x.view(x.size()[0], -1)
        h = self.fc1(x_)
        l1 = torch.norm(self.fc1.weight, p=1)
        return h, l1
        