#import sys
import gym
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from ddpg.ddpg_ddpg import DDPGagent
from ddpg.ddpg_utils import *
from environment_dqn_n import *

env = Production_DQN_Env()

agent = DDPGagent(env)
noise = GaussianNoise(env.action_space)
batch_size = 16
rewards = []
avg_rewards = []

for episode in range(10):
    print("Episode:", episode)
    state, _ = env.reset()
    #noise.reset()
    episode_reward = 0
    
    step = 0
    done = False
    while not done:
        print("\tStep:", step)
        step += 1
        action = agent.get_action(state)
        action = noise.get_action(action)
        #print("\tAction", action)
        new_state, reward, done, _, _ = env.step(action) 
        agent.memory.push(state, action, reward, new_state, done)
        
        if len(agent.memory) > batch_size:
            agent.update(batch_size)        
        
        state = new_state
        episode_reward += reward

        if done:
            print("episode: {}, reward: {}, average _reward: {} \n".format(episode, np.round(episode_reward, decimals=2), np.mean(rewards[-10:])))
            break

    rewards.append(episode_reward)
    avg_rewards.append(np.mean(rewards[-10:]))

plt.plot(rewards)
plt.plot(avg_rewards)
plt.plot()
plt.xlabel('Episode')
plt.ylabel('Reward')
plt.show()