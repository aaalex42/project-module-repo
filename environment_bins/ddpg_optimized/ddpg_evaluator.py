"""
Reference: https://github.com/ghliu/pytorch-ddpg/blob/master/evaluator.py
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