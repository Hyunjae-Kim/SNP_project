import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, snp_len, n_channel, k_size, s_size):
        super(Model, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, n_channel, kernel_size=(3,k_size), stride=(1,s_size)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm2d(n_channel))
        self.out1 = int(snp_len/s_size)-1
        self.conv2 = nn.Sequential(
            nn.Conv1d(n_channel, n_channel, kernel_size=k_size, stride=s_size),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(n_channel))
        self.out2 = int(self.out1/s_size)-1
        self.conv3 = nn.Sequential(
            nn.Conv1d(n_channel, n_channel, kernel_size=int(k_size/2), stride=int(s_size/2)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(n_channel))
        self.out3 = int(self.out2/int(s_size/2))-1
        self.conv4 = nn.Sequential(
            nn.Conv1d(n_channel, n_channel, kernel_size=int(k_size/2), stride=int(s_size/2)),
            nn.LeakyReLU(negative_slope=0.1),
            nn.BatchNorm1d(n_channel))
        self.out4 = int(self.out3/int(s_size/2))-1
        self.fc1 = nn.Linear(n_channel*self.out4, 1)

    def forward(self, data, target):
        x = self.conv1(data)
        x = x.view(x.size()[0], x.size()[1],-1)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x_ = x.view(x.size()[0], -1)
        h = self.fc1(x_)
        l1 = torch.norm(self.fc1.weight, p=1)
        return h, l1
        