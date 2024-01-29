"""
TODO:
    - Change the output layers to have the same format as action and/or observation spaces
"""

import torch
import torch.nn as nn
import torch.nn.functional as F 
import torch.autograd
from torch.autograd import Variable
from init_vars import *


class Critic(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(Critic, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, state, action):
        """
        Params state and actions are torch tensors
        """
        x = torch.cat([state, action], 1)
        #x = x.to(torch.float32)
        x = self.sequential(x)

        return x

class Actor(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, learning_rate = 3e-4):
        super(Actor, self).__init__()
        self.sequential = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

        self.sigmoid = nn.Sigmoid()
        self.sigmoid_2 = nn.Sigmoid()
        self.relu = nn.ReLU()
        
    def forward(self, state):
        """
        Param state is a torch tensor
        """
        x = self.sequential(state)

        #print("x", x)

        p = torch.round(self.sigmoid(x[:, 0])).unsqueeze(0)
        qty = self.sigmoid_2(x[:, 1])
        #print("qty_0", qty)
        qty = ((MAXIMUM_INVENTORY // BIN_SIZE) * qty).unsqueeze(0) #torch.clamp(self.relu(x[:, 1]), max=MAXIMUM_INVENTORY // BIN_SIZE).unsqueeze(0)

        #print("qty", qty)

        return torch.cat((p, qty), 0).transpose(0, 1)