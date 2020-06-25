import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils as utils
import torch.nn.init as init

class Model(nn.Module):
    def __init__(self, input_size, attend_size, hidden_size, seq_len):
        super(Model, self).__init__()
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        
        self.mat_Q = nn.Linear(input_size, attend_size)
        self.mat_K = nn.Linear(input_size, attend_size)
        self.mat_V = nn.Linear(input_size, hidden_size)
        self.softmax = nn.Softmax(dim=-1)
        self.fc1 = nn.Linear(hidden_size*seq_len, 1)

    def forward(self, data, target):
        q_mat = self.mat_Q(data)
        k_mat = self.mat_K(data).transpose(1,2)
        v_mat = self.mat_V(data)
        attend_score = self.softmax(torch.matmul(q_mat,k_mat))
        
        h = torch.matmul(attend_score, v_mat).view(-1, self.hidden_size*self.seq_len)
        h = self.fc1(h)
        l1 = torch.norm(self.fc1.weight, p=1)
        return h, l1
        