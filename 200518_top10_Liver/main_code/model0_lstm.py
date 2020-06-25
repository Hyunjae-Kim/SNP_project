import numpy as np
import module

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, t_step):
        super(Model, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, \
                    batch_first=True, bidirectional=True)
        self.fc1 = nn.Sequential(nn.Linear(hidden_size*2, 1),
                                 nn.LeakyReLU())                            
        self.fc2 = nn.Linear(t_step, 1)
    
    def loss_func(self, dX, dY):
        loss = torch.mean((dX-dY)**2)
        return loss

    def forward(self, data, target):
        x, (hid,c) = self.rnn(data)
        x = self.fc1(x)
        x = x.view(x.size()[0],-1)
        h = self.fc2(x)
        
        loss = self.loss_func(h, target)
        return loss, h
        