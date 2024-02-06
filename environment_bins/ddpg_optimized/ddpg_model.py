import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from init_vars import *


#Dictonary of activation functions
activationDict = {
    "relu" : nn.ReLU(),
    "elu"  : nn.ELU(),
    "tanh" : nn.Tanh(),
    "leakyrelu" : nn.LeakyReLU(),
    "sigmoid"   : nn.Sigmoid()
}


def create_mlp_model(hidden_dims, dropout = None, batchnorm = False, activation = "relu"):
    """
    creates a mlp model with given parameters

    Parameters:
    hidden_dims:    list of integers, len of list is number of layers and value of element in list is number of neurons in that layer
    dropout:        list of floats, len must be len(hidden_dims)-2, values of elements must be between 0 and 1, value is dropout rate per hidden layer
    batchnorm:      bool, True = adding normalization layers, False = no normalization layers
    activation:     list of string, possible activation functions are in activatinoDict, Relu, Elu, tanh, leakyRelu, sigmoid

    Return:
    container with the created model

    droupout and batchnorm are not implemented in Actor and Critic classes
    """
    if dropout == None:
        dropout = [0] * (len(hidden_dims) - 2)
    modules = [activationDict[activation]]
    for index in range(len(hidden_dims) - 1):
        modules.append(nn.Linear(in_features=hidden_dims[index], out_features=hidden_dims[index + 1], bias=True))
        if index != len(hidden_dims) - 2:
            modules.append(activationDict[activation]) #add activation layer
            if batchnorm == True: #add batchnorm layers
                modules.append(nn.BatchNorm1d(hidden_dims[index + 1], affine=False))
            if dropout[index] > 0: #add dropout layers
                modules.append(nn.Dropout(p=dropout[index]))
    
    modules.append(activationDict[activation])
    return modules


def initialize_weights(model, init_w):
    """
    initializes the weights of the model
    """
    if isinstance(model, nn.Linear):
        nn.init.uniform_(model.weight, -init_w, init_w)
        return model

    for idx, layer in enumerate(model):
        if isinstance(layer, nn.Linear):
            v = 1 / np.sqrt(layer.weight.data.size()[0])
            nn.init.uniform_(layer.weight, -v, v)
            nn.init.constant_(layer.bias, 0)

    return model


class Actor(nn.Module):
    """
    Decides which action should be taken in a given state.
    """
    def __init__(self, nb_states, hidden_layer_list = [64, 128, 32], init_w = 3e-3) -> None:
        super().__init__()
        self.model = create_mlp_model(hidden_dims=hidden_layer_list, dropout=None, batchnorm=False, activation="relu")
        self.model.insert(0, nn.Linear(nb_states, hidden_layer_list[0]))
        self.model = nn.Sequential(*self.model)
        self.linear_act = nn.Linear(hidden_layer_list[-1], 2) # number of products
        self.linear_bin = nn.Linear(hidden_layer_list[-1], 1) # predicting the bin
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

        self.init_w = init_w
        self.init_weights()
    
    def init_weights(self):
        self.model = initialize_weights(self.model, self.init_w)
        self.linear_act = initialize_weights(self.linear_act, self.init_w)
        self.linear_bin = initialize_weights(self.linear_bin, self.init_w)
    
    def forward(self, x):
        out = self.model(x)

        act = self.linear_act(out)
        act = self.sigmoid(act)

        bin = self.linear_bin(out)
        bin = self.relu(bin)

        ##print("BEFORE", "act", act.shape, "\nbin", bin.shape)

        act = torch.argmax(act, dim = 1).unsqueeze(0).transpose(0, 1)
        bin = torch.clamp(bin, 0, MAXIMUM_INVENTORY // BIN_SIZE)

        ##print("AFTER", "act", act.shape, "\nbin", bin.shape)

        action = torch.cat([act, bin], dim=1)
        action = torch.round(action)

        ##print("ACTION", action.shape)

        return action


class Critic(nn.Module):
    """
    Evaluates the action taken by the Actor, informs it of the quality of the action 
    and how it should adjust.

    Done differently than in the original GitHub repo.
    """
    def __init__(self, nb_states, nb_actions, hidden_layer_list = [64, 128, 32], init_w = 3e-3) -> None:
        super().__init__()
        self.model = create_mlp_model(hidden_dims=hidden_layer_list, dropout=None, batchnorm=False, activation="relu")
        self.model.insert(0, nn.Linear(nb_states + nb_actions, hidden_layer_list[0]))
        self.model.append(nn.Linear(hidden_layer_list[-1], 1))
        self.model = nn.Sequential(*self.model)

        self.init_w = init_w
        self.init_weights()
    
    def init_weights(self):
        self.model = initialize_weights(self.model, self.init_w)
    
    def forward(self, x):
        return self.model(x)