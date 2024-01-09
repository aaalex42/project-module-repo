import gym 
from gym import spaces
from main import *
from classes_n import *


class Production_DDPG_Env(gym.Env):
    """
    This class is a wrapper for the production classes according to the OpenAI Gym rules.
    It is based on discrete action and observation spaces.
    """
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([2, 300_000])
        self.observation_space = None

        #initialize the demands, inventories, warehouse and machine
        self.machine = Machine(
            Warehouse(
                Inventory(Demand(P1)),
                Inventory(Demand(P2))
            )
        )


    def step(self, action):
        assert self.action_space.contains(action)



    def _get_obs(self):
        pass


    def reset(self):
        pass


    def render(self):
        pass


    def close(self):
        pass