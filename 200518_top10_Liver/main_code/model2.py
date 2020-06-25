import numpy as np
import module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, snp_len, alpha):
        super(Model, self).__init__()
        self.alpha = alpha
        self.fc1 = nn.Linear(snp_len, 1)
    
    def loss_func(self, dX, dY):
        loss = torch.mean((dX-dY)**2)
        return loss

    def forward(self, data, target):
        h = self.fc1(data)
        l = self.loss_func(h, target)
        l1 = torch.norm(self.fc1.weight, p=1)
        loss = l+ self.alpha*l1
        return loss, h
        