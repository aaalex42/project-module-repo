"""
!!!THE CODE IS NOT USED ANYMORE AND IT IS NOT WORKING!!!

Check if round and clamp are ok to use in the forward pass.
"""

import torch
from torch import nn
import copy
from init_vars import *

class AgentDDQNet(nn.Module):
    """
    A neural network for the DDQ agent.
    The hidden layers can be later adapted for the best performance.
    It has 2 outputs. The first one predicts which product to produce (o or 1).
    The second one predicts the quantity to produce (0 to MAXIMUM_INVENTORY // BIN_SIZE).
    """
    def __init__(self, input_dim):
        super(AgentDDQNet, self).__init__()

        self.online = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2)
        )

        self.target = copy.deepcopy(self.online)

        # Q_target parameters are frozen
        for p in self.target.parameters():
            p.requires_grad = False
        
    
    def forward(self, x, model):
        if model == 'online':
            p = torch.round(nn.Sigmoid(self.online[:, 0]))
            qty = torch.clamp(nn.ReLU(self.online[:, 1]), max=MAXIMUM_INVENTORY // BIN_SIZE)
            return p, qty
        elif model == 'target':
            p = torch.round(nn.Sigmoid(self.target[:, 0]))
            qty = torch.clamp(nn.ReLU(self.target[:, 1]), max=MAXIMUM_INVENTORY // BIN_SIZE)
            return p, qty
        else:
            raise ValueError(f'{model} is not a valid model name')