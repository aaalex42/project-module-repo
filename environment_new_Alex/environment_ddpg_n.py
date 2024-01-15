import gym 
from gym import spaces
from main import *
from classes_n import *


class Production_DDPG_Env(gym.Env):
    """
    This class is a wrapper for the production classes according to the OpenAI Gym rules.
    It is based on discrete action and observation spaces.

    ### Action space:
        - id0: 0 or 1: product 1 or 2
        - id1: 0 to SOME_QUANTITY: quantity to produce
    
        | Num | Action   | Min   | Max               | Type     |
        |-----|----------|-------|-------------------|----------|
        | 0   | Product  | 0     | 1                 | Discrete |
        | 1   | Quantity | 0     | MAXIMUM_INVENTORY | Discrete |

    ### Observation space:
        The observation is a `ndarray` with shape `(4,)`.

        - id0: 0 to MAXIMUM_INVENTORY: quantity of product 1 in inventory
        - id1: 0 to MAXIMUM_INVENTORY: quantity of product 2 in inventory
        - id2: 0 or 1: for which product an order is placed
        - id3: 0 to MAXIMUM_INVENTORY / 2: order amount to be delivered (demand)
    
        | Num | Observation      | Min  | Max                   | Type     |
        |-----|------------------|------|-----------------------|----------|
        | 0   | Inventory P1     | 0    | MAXIMUM_INVENTORY     | Discrete |
        | 1   | Inventory P2     | 0    | MAXIMUM_INVENTORY     | Discrete |
        | 2   | Order product    | 0    | 1                     | Discrete |
        | 3   | Order quantity   | 0    | MAXIMUM_INVENTORY / 2 | Discrete |

    ### Reward:
        The reward function is defined as:
        R = 1 - current_warehouse_level / MAXIMUM_INVENTORY

    ### Starting state:

    ### Episode truncation:

    ### Arguments:
        All arguments are given in main.py file.

    ### Version history:
        No version history yet.
    """
    def __init__(self):
        self.action_space = spaces.MultiDiscrete([2, MAXIMUM_INVENTORY])
        self.observation_space = spaces.MultiDiscrete([MAXIMUM_INVENTORY, MAXIMUM_INVENTORY, 2, MAXIMUM_INVENTORY / 2])

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