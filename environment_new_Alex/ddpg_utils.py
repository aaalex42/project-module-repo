import numpy as np
import gym
from collections import deque
import random

# Gaussian noise
class GaussianNoise:
    def __init__(self, action_space, mu = 0, sigma = 5) -> None: 
        # a big sigma is necessary for rounding to integer after predicting
        self.mu = mu
        self.sigma = sigma
        #self.action_dim = action_space.shape[0] #not necessary if the agent can only produce a product in a day
        self.action_space = action_space

    def get_action(self, action):
        noise = np.random.normal(self.mu, self.sigma)
        return np.array(
            [action[0], np.round(np.clip(action[1] + noise, 0, self.action_space[1].n)) ]
        )
        

class Memory:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)
    
    def push(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        
        return state_batch, action_batch, reward_batch, next_state_batch, done_batch

    def __len__(self):
        return len(self.buffer)