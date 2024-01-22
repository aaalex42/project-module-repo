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
        | 2   | Demand p1        | 0    | MAXIMUM_INVENTORY / 2 | Discrete |
        | 3   | Demand p2        | 0    | MAXIMUM_INVENTORY / 2 | Discrete |

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
        self.observation_space = spaces.MultiDiscrete([MAXIMUM_INVENTORY, MAXIMUM_INVENTORY, MAXIMUM_INVENTORY / 2, MAXIMUM_INVENTORY / 2])

        #initialize the demands, inventories, warehouse and machine
        self.machine = Machine(
            Warehouse(
                Inventory(Demand(P1)),
                Inventory(Demand(P2))
            )
        )


    def step(self, action): 
        assert self.action_space.contains(action)

        terminated = False

        #check if maximum warehouse level is exceeded, or inventory is negative
        #for both cases, the game ends with negative reward
        """CHECK IF IT IS EVEN POSSIBLE TO HAVE NEGATIVE INVENTORY OR EXCEED MAXIMUM WAREHOUSE LEVEL"""
        if self.machine.warehouse.current_warehouse_level > MAXIMUM_INVENTORY   \
                or self.machine.warehouse.products[0].inventory_level[-1] < 0   \
                or self.machine.warehouse.products[1].inventory_level[-1] < 0:
            #GIVE NEGATIVE REWARD (if necessary) and RESTART THE GAME
            reward = -1
            terminated = True


        exit_code_prod = self.machine.produce(action)
        exit_code_fulf = self.machine.fulfill()
        #check if an order can be produced and stored
        if exit_code_prod == 0:
            reward, terminated = self.check_fulfill(exit_code_fulf)
        
        elif exit_code_prod == 101: # the order can be produced, but not stored
            reward = -1
            terminated = True #meaning that the maximum warehouse level is exceeded
        
        else: 
            #for the cases when exit code for production is 201
            reward, terminated = self.check_fulfill(exit_code_fulf)

        #return observation, reward, terminated, False, {}
        return self._get_obs(), reward, terminated, False, {}


    def _get_obs(self):
        return (self.machine.warehouse.products[0].inventory_level[-1],
                self.machine.warehouse.products[1].inventory_level[-1], 
                self.machine.warehouse.products[0].demand_class.demand[self.machine.t],
                self.machine.warehouse.products[1].demand_class.demand[self.machine.t]
                )


    def reset(self, seed = None, options = None):
        super().reset(seed = seed)

        #reset the demands, inventories, warehouse and machine
        self.machine = Machine(
            Warehouse(
                Inventory(Demand(P1)),
                Inventory(Demand(P2))
            )
        )

        return self._get_obs(), {}


    def render(self):
        pass


    def close(self):
        pass


    def check_fulfill(self, exit_code):
        terminated = False
        #check if an order can be fulfilled
        if exit_code == 11:
            #give positive reward
            reward = 1 - self.machine.warehouse.current_warehouse_level / MAXIMUM_INVENTORY
        
        # if there is no demand
        elif exit_code == 10:
            reward = 0
        
        # if the order cannot be fulfilled
        elif exit_code == 12: #or just else
            reward = -1
            # HERE A GAME SHOULD END
            terminated = True
        
        return reward, terminated