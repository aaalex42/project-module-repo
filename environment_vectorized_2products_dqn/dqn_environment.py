import numpy as np
from classes_n import Inventory, Demand, Warehouse, Machine
from main import P1, P2
import gym
from gym import spaces

class Production_DQN_Env(gym.Env):
    """
    This class is a wrapper for the production classes according to the OpenAI Gym rules.
    It is based on discrete action and observation spaces.
    """
    def __init__(self):
        super(Production_DQN_Env, self).__init__()
        
        #class attributes
        self.action_space = spaces.MultiDiscrete([30000,100000]) #order amount product1 = 30.000 pieces, order amount product2 = 100.000 pieces
        #self.observation_space = spaces.Tuple((spaces.Discrete(n=2,start=0), spaces.Box(low=0, high=300000, dtype=np.int32)))
        self.observation_space = spaces.Dict({
            'p1_inventory': spaces.Discrete(300000), 
            'p2_inventory': spaces.Discrete(300000),  
            'p1_demand': spaces.Discrete(300000),  
            'p2_demand': spaces.Discrete(300000),  
        })

        #initialize the demands, inventories, warehouse and machine
        self.machine = Machine(
            Warehouse(
                Inventory(Demand(P1)),
                Inventory(Demand(P2))
            )
        )

    def step(self, action):
        """
        Executes one time step within the environment.
        """
        #check if action is valid
        assert self.action_space.contains(action)            
        
        #execute order
        self.machine.produce(action)

        #fulfill order
        self.machine.fulfill()
       
        #get exit codes
        exit_codes = self._get_exit_codes(action)

        #Reward function
        if exit_codes[0] == 0:
            reward = 1
        elif exit_codes[0] == 101:
            reward = 0
        elif exit_codes[0] == 210:
            reward = -1
            
        elif exit_codes[1][0] or exit_codes[1][1] == 11:
            reward = 1

        elif exit_codes[1][0] or exit_codes[1][1] == 12:
            reward = -1
        
        else:
            reward = 0  

        #get observation
        observation = self._get_obs()

        #check if done
        done = False

        #Print observation and reward (for debugging)
        print(observation, reward)

        return observation, reward, done, {}
    
    def _get_obs(self):
        """
        Returns the current observation.
        """
        #get observation for inventory
        p1_inventory = self.machine.warehouse.products[0].inventory_level[0]
        p2_inventory = self.machine.warehouse.products[1].inventory_level[1]

        #get observation for demand
        p1_demand = self.machine.warehouse.products[0].demand_class.demand[0]
        p2_demand = self.machine.warehouse.products[1].demand_class.demand[1]

        return p1_inventory, p2_inventory, p1_demand, p2_demand

    def _get_exit_codes(self,action):
        exit_code_produce = self.machine.produce(action)

        exit_code_fulfill = self.machine.fulfill()

        return exit_code_produce, exit_code_fulfill

    def reset(self):

        #initialize the demands, inventories, warehouse and machine
        self.machine = Machine(
            Warehouse(
                Inventory(Demand(P1)),
                Inventory(Demand(P2))
            )
        )
        pass


    def render(self):
        pass


    def close(self):
        pass