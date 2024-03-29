import gymnasium as gym
import numpy as np
import math
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from dqn_utils import *
from init_vars import *

# set up matplotlib
is_ipython = 'inline' in matplotlib.get_backend()
if is_ipython:
    from IPython import display

plt.ion()

# if GPU is to be used
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device is set to: ", device)


# Transition is a named tuple representing a single transition in our environment.
Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    """A cyclic buffer of bounded size that holds the transitions observed recently."""
    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity) 

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)
    
class DQN(nn.Module):
    """A simple fully connected neural network with 3 layers"""

    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 128)
        self.layer2 = nn.Linear(128, 128)

        # Output layer for product ID
        self.out_product = nn.Linear(128, 2)

        # Output layer for order amount
        self.out_order = nn.Linear(128, 1)

    # Called with either one element to determine next action, or a batch
    # during optimization. Returns tensor([[left0exp,right0exp]...]).
    def forward(self, x):
        # Normalize the input data
        mean = torch.mean(x)
        std = torch.std(x)
        x = (x - mean) / std

        
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))

        

        out_product = self.out_product(x)
        out_product = torch.sigmoid(out_product) #sigmoid function to get values between 0 and 1 (Product 1 or Product 2)

        out_order = self.out_order(x) #no activation function to get values between -inf and +inf (order amount)
        return out_product, out_order


# BATCH_SIZE is the number of transitions sampled from the replay buffer
# GAMMA is the discount factor as mentioned in the previous section
# EPS_START is the starting value of epsilon
# EPS_END is the final value of epsilon
# EPS_DECAY controls the rate of exponential decay of epsilon, higher means a slower decay
# TAU is the update rate of the target network
# LR is the learning rate of the ``AdamW`` optimizer
    
class DQNAgent:
    def __init__(self, env, device, num_episodes=50, BATCH_SIZE=64, BATCH_START_SIZE=10000, MEMORY_SIZE=100000, GAMMA=0.99, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=1000, TAU=0.005, LR=1e-5):
        self.BATCH_SIZE = BATCH_SIZE
        self.BATCH_START_SIZE = BATCH_START_SIZE
        self.MEMORY_SIZE = MEMORY_SIZE
        self.GAMMA = GAMMA
        self.EPS_START = EPS_START
        self.EPS_END = EPS_END
        self.EPS_DECAY = EPS_DECAY
        self.TAU = TAU
        self.LR = LR
        self.env = env
        self.device = device
        self.num_episodes = num_episodes

        # Get the maximum order amount from the gym env
        self.max_order_amount = self.env.action_space.nvec[1] - 1 #the maximum order amount is the maximum value of the action space minus 1

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.shape[0]

        # Get the number of observations from the gym env
        state, info = self.env.reset()
        
        #self.n_observations = len(state)
        self.n_observations = self.env.observation_space.shape[0]
       
        self.policy_net = DQN(self.n_observations, self.n_actions).to(device) #the policy network is the network that is being trained
        self.target_net = DQN(self.n_observations, self.n_actions).to(device) #the target network is the network that is used to calculate the target values  
        self.target_net.load_state_dict(self.policy_net.state_dict()) #initialize the target network with the same weights as the policy network

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.LR, amsgrad=True)
        self.memory = ReplayMemory(self.MEMORY_SIZE)

        self.steps_done = 0
        self.episode_durations = []
        self.total_reward = 0

        #for plotting
        self.inventory_levels = []
        self.total_rewards = []

    def select_action(self, state):
        #global env.machine.t 
        sample = random.random() 
        eps_threshold = self.EPS_END + (self.EPS_START - self.EPS_END) * \
            math.exp(-1. * self.env.machine.t / self.EPS_DECAY)
        #print('eps_threshold:', eps_threshold)
        if sample > eps_threshold:
            with torch.no_grad(): 
                out_product, out_order = self.policy_net(state)                
                product_id = out_product.max(1).indices.view(1, 1)     
                order_amount = (torch.sigmoid(out_order) * self.max_order_amount).view(1,1).round().long() 
                action = torch.cat((product_id, order_amount), dim=1)
                return action.squeeze() 
        else:
            return torch.tensor(self.env.action_space.sample(), device=self.device, dtype=torch.long)

    def plot_durations(self, show_result=False):
        plt.figure(1)
        durations_t = torch.tensor(self.episode_durations, dtype=torch.float)
        if show_result:
            plt.title('Result')
        else:
            plt.clf()
            plt.title('Training...')
        plt.xlabel('Episode')
        plt.ylabel('Duration')
        plt.plot(durations_t.numpy())
    
        # Take 100 episode averages and plot them too
        if len(durations_t) >= 100:
            means = durations_t.unfold(0, 100, 1).mean(1).view(-1)
            means = torch.cat((torch.zeros(99), means))
            plt.plot(means.numpy())

        plt.pause(0.001)  # pause a bit so that plots are updated
        if is_ipython:
            if not show_result:
                display.display(plt.gcf())
                display.clear_output(wait=True)
            else:
                display.display(plt.gcf())



    def optimize_model(self):
        if len(self.memory) < self.BATCH_START_SIZE: 
            return
        transitions = self.memory.sample(self.BATCH_SIZE)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        #print(batch.next_state)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action).view(-1, 2, 1) #same dimension as the output of the network
        action_batch = action_batch[:, :1, :] #use only the first column is the product id to get same dimension as the output of the network	
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        
        out_product, out_order = self.policy_net(state_batch)

        # Ensure out_product and out_order have the same size in the first dimension        
        out_order = out_order.expand(-1, out_product.size(1)).clamp(0, self.max_order_amount)        

        out = torch.cat((out_product.unsqueeze(-1), out_order.unsqueeze(-1)), dim=-1) 
        #out = torch.cat((out_product, out_order), dim=-1)
        # Assuming action_batch is a [BATCH_SIZE x 2] tensor where the first column
        # contains the indices for out_product and the second column contains the indices for out_order
        state_action_values = out.gather(2, action_batch).squeeze(-1) 
        #state_action_values = out.gather(1, action_batch.unsqueeze(-1)).squeeze(-1)
        
        #state_action_values = self.policy_net(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1).values
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.BATCH_SIZE, device=device)
        with torch.no_grad():
            out_product, out_order = self.target_net(non_final_next_states)
            next_state_values[non_final_mask] = out_product.max(1).values
            #next_state_values[non_final_mask] = self.target_net(non_final_next_states).max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.GAMMA) + reward_batch

        # Compute Huber loss
        criterion = nn.CrossEntropyLoss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)    
        self.optimizer.step()



    def train(self):
        if torch.cuda.is_available():
            print("Using GPU")
            num_episodes = self.num_episodes
        else:
            num_episodes = 50
        
        

        for i_episode in range(num_episodes):   
          
            # Initialize the environment and get it's state
            state, info = self.env.reset()
            state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            for t in count():
                action = self.select_action(state)              
                observation, reward, terminated,  _, _ = self.env.step(action)     
                self.env.inc_t()                       
                self.total_reward += reward
                reward = torch.tensor([reward], device=device)                
                self.inventory_levels.append(observation[0] + observation[1]) #Correct?
                self.total_rewards.append(self.total_reward)        
                done = terminated 
                if done:
                    self.env.reset()
                    next_state = None
                else:
                    next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)

                # Store the transition in memory                
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()

                # Soft update of the target network's weights
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key]*self.TAU + target_net_state_dict[key]*(1-self.TAU)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    self.episode_durations.append(t + 1)
                    break
                 
            print("Episode: ", i_episode,'|' ," Duration: ", t+1, '|' ,"Total Reward:", self.total_reward, '|' ,'Terminated:', terminated)   
        return self.total_rewards, self.inventory_levels, self.episode_durations
    
    def plot_rewards(self, total_rewards, inventory_levels):
        fig, ax1 = plt.subplots()

        color1 = 'tab:blue'
        color2 = 'tab:green'

        ax1.set_xlabel('Simulation Steps')
        ax1.set_ylabel('Inventory Level', color=color1)
        ax1.plot(inventory_levels, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Total Reward', color=color2)
        ax2.plot(total_rewards, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()


        print("total_rewards: ", total_rewards[-1])
        print('mean reward per episode is: ', np.mean(total_rewards))
        print("Complete")
        print('Servicelevel not met for Product 1:', self.env.service_level_not_met_1, '|', 'Servicelevel not met for Product 2:', self.env.service_level_not_met_2)
        print('Total Average inventory level :', np.mean(inventory_levels))
        self.plot_durations(show_result=True)
        plt.ioff()
        plt.show()
