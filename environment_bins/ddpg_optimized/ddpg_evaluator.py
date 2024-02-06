"""
Reference: https://github.com/ghliu/pytorch-ddpg/blob/master/evaluator.py

TODO:
    - Add the option to switch easily between SkipStep Env and the original one!
        This is also for displaying the episode length in days instead of hours (or skip steps).
    - Add the option to include a super class for accessing args and kwargs for different RL algs.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.io import savemat

from ddpg_optimized.ddpg_utils import *


class Evaluator(object):
    def __init__(self, num_episodes, interval, save_path = "", max_episode_length = None) -> None:
        self.num_episodes = num_episodes
        self.max_episode_length = max_episode_length
        self.interval = interval
        self.save_path = save_path
        self.results = np.array([]).reshape(num_episodes, 0)

    def __call__(self, env, policy, debug = False, save = True):
        self.is_training = False
        observation = None
        result = []

        for episode in range(self.num_episodes):
            #reset the environment at the start of episode
            observation, _ = env.reset()
            episode_steps = 0
            episode_reward = 0

            assert observation is not None

            #start the episode
            done = False
            while not done: 
                #get an action from policy
                action = policy(observation).astype(int)

                #take a step
                observation, reward, done, _, _ = env.step(action)
                if self.max_episode_length and episode_steps >= self.max_episode_length - 1:
                    done = True
                
                #update the result
                episode_reward += reward
                episode_steps += 1
            
            if debug:
                prYellow("[EVALUATE] #Episode {} - Length: {} - Reward: {}".format(episode, episode_steps, episode_reward))
            result.append(episode_reward)

        result = np.array(result).reshape(-1, 1)
        self.results = np.hstack([self.results, result])

        if save:
            self.save_results("{}/validate_reward".format(self.save_path))
        return np.mean(result)

    def save_results(self, filename):
        #x for steps
        x_axis = range(0, self.results.shape[1] * self.interval, self.interval)
        #y
        y_mean = np.mean(self.results, axis = 0)
        #error bar
        std = np.std(self.results, axis = 0)
        #plotting the rewards
        fig, ax = plt.subplots(1, 1, figsize = (8, 6))
        ax.errorbar(x_axis, y_mean, yerr = std, fmt = "-o") #capsize = 5, linewidnth = 2, elinewidth = 1 or other values
        plt.xlabel("Time Step")
        plt.ylabel("Average Reward")
        plt.savefig(filename + ".png")
        savemat(filename + ".mat", {"reward": self.results})

