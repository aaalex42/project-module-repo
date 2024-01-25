import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = []
        self.gamma = 0.99  # discount factor
        self.epsilon = 1.0  # exploration rate
        self.epsilon_decay = 0.995
        self.epsilon_min = 0.01
        self.model = self._build_model() #it is a neural network that takes the state as input and outputs the Q-values for each action in the action space.

    def _build_model(self):
        #build model
        pass
        

    def remember(self, state, action, reward, next_state, done):            
        pass

    def act(self, state):
        pass

    def replay(self, batch_size):    
        pass