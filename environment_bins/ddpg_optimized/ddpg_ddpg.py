"""
TODO:
    - Change the volatile method, since it is deprecated and it does not make sense now
"""


import numpy as np
import torch
import torch.nn as nn   
from torch.optim import Adam
from collections import namedtuple

from ddpg_optimized.ddpg_model import Actor, Critic
from ddpg_optimized.ddpg_memory import SequentialMemory
from ddpg_optimized.ddpg_random_process import GaussianWhiteNoiseProcess
from ddpg_optimized.ddpg_utils import *
from init_vars import MAXIMUM_INVENTORY, BIN_SIZE

criterion = nn.MSELoss()

class DDPG_agent(object):
    def __init__(self, nb_states, nb_actions, args : namedtuple) -> None:
        if args.seed > 0:
            self.seed(args.seed)

        self.nb_states = nb_states
        self.nb_actions = nb_actions

        network_config = {
            "hidden_layer_list": args.hidden_layer_list,
            "init_w": args.init_w
        }
        self.actor = Actor(self.nb_states, **network_config)
        self.actor_target = Actor(self.nb_states, **network_config)
        self.actor_optim = Adam(self.actor.parameters(), lr=args.lr_actor)

        self.critic = Critic(self.nb_states, self.nb_actions, **network_config)
        self.critic_target = Critic(self.nb_states, self.nb_actions, **network_config)
        self.critic_optim = Adam(self.critic.parameters(), lr=args.lr_critic)

        # initialize targets same as original networks
        hard_update(self.actor_target, self.actor)
        hard_update(self.critic_target, self.critic)

        #create the replay buffer and random process
        self.memory = SequentialMemory(limit=args.memory_size, window_length=args.window_length)
        """CHECK THIS PART!!!"""
        self.random_process = GaussianWhiteNoiseProcess(mu=args.noise_mu, sigma=args.noise_sigma)

        # initialize hyper-parameters
        self.batch_size = args.batch_size
        self.tau = args.tau
        self.discount = args.discount
        self.depsilon = 1.0 / args.epsilon_decay

        self.epsilon = 1.0
        self.cur_state = None
        self.cur_action = None
        self.is_training = True

        if USE_CUDA: self.cuda()
    
    def update_policy(self):
        # sample batch
        state_batch, action_batch, reward_batch, \
        next_state_batch, terminal_batch = self.memory.sample_and_split(self.batch_size)

        # prepare the batch for the Q target
        ###print("actor target", self.actor_target(to_tensor(next_state_batch, volatile=True)))
        ###print("DONEEEEEEE")
        input_critic_target = torch.cat(
            (
                to_tensor(next_state_batch, volatile=True),
                self.actor_target(to_tensor(next_state_batch, volatile=True))
            ), dim = 1
        )
        with torch.no_grad():
            next_q_values = self.critic_target(input_critic_target)
        #next_q_values.volatile = False

        target_q_batch = to_tensor(reward_batch) + \
            self.discount * to_tensor(terminal_batch.astype(float)) * next_q_values
        
        # Critic update
        self.critic.zero_grad()

        input_critic = torch.cat(
            (
                to_tensor(state_batch),
                to_tensor(action_batch)
            ), dim = 1
        )
        q_batch = self.critic(input_critic)
        
        value_loss = criterion(q_batch, target_q_batch)
        value_loss.backward()
        self.critic_optim.step()

        # Actor update
        self.actor.zero_grad()

        input_policy = torch.cat(
            (
                to_tensor(state_batch),
                self.actor(to_tensor(state_batch))
            ), dim = 1
        )
        policy_loss = -self.critic(input_policy)

        policy_loss = policy_loss.mean()
        policy_loss.backward()
        self.actor_optim.step()

        # Target update
        soft_update(self.actor_target, self.actor, self.tau)
        soft_update(self.critic_target, self.critic, self.tau)
    
    def eval(self):
        self.actor.eval()
        self.actor_target.eval()
        self.critic.eval()
        self.critic_target.eval()
    
    def cuda(self):
        self.actor.cuda()
        self.actor_target.cuda()
        self.critic.cuda()
        self.critic_target.cuda()
    
    def observe(self, cur_reward, new_state, done):
        if self.is_training:
            self.memory.append(self.cur_state, self.cur_action, cur_reward, done)
            self.cur_state = new_state

    def random_action(self):
        action_product = np.random.choice([0, 1])
        action_qty = np.random.randint(0, MAXIMUM_INVENTORY // BIN_SIZE)
        self.cur_action = (action_product, action_qty)
        return (action_product, action_qty)
    
    def select_action(self, cur_state, decay_epsilon=True):
        action = self.actor(to_tensor(np.array([cur_state])))
        """action_product = torch.argmax(action[0], dim = 1).unsqueeze(0)
        action = torch.cat([action_product, action[1]], dim=1)
        action = torch.round(action)"""
        action = to_numpy(
            action
        ).squeeze(0)
        action[1] += self.is_training*max(self.epsilon, 0)*self.random_process.sample()
        action[1] = np.clip(action[1], 0., MAXIMUM_INVENTORY // BIN_SIZE)

        if decay_epsilon:
            self.epsilon -= self.depsilon
        
        action = action.astype(int)
        self.cur_action = action
        return action
    
    def reset(self, obs):
        self.cur_state = obs
        self.random_process.reset_states()

    def load_weights(self, output):
        if output is None: return

        self.actor.load_state_dict(
            torch.load('{}/actor.pkl'.format(output))
        )

        self.critic.load_state_dict(
            torch.load('{}/critic.pkl'.format(output))
        )
    
    def save_model(self,output):
        torch.save(
            self.actor.state_dict(),
            '{}/actor.pkl'.format(output)
        )
        torch.save(
            self.critic.state_dict(),
            '{}/critic.pkl'.format(output)
        )
    
    def seed(self, s):
        torch.manual_seed(s)
        if USE_CUDA:
            torch.cuda.manual_seed(s)
