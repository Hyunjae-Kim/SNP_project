import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, t_step):
        super(Model, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, \
                    batch_first=True, bidirectional=False)
        self.fc1 = nn.Linear(t_step, 1)

    def forward(self, data, target):
        x, (hid,c) = self.rnn(data)
        x = x[:,:,-1]
        h = self.fc1(x)
        l1 = torch.norm(self.fc1.weight, p=1) 
        return h, l1