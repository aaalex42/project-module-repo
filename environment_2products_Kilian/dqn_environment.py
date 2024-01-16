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

        #get new demand
        
        #production order
        
        
        #execute order
        self.machine.produce(action)

        #fulfill order
        self.machine.fulfill(action)
        
        #get reward
        if self.machine.warehouse.inventory_1.demand.fulfilled == 10 or 11 and self.machine.warehouse.inventory_2.demand.fulfilled == 10 or 11:
            reward = 1
        elif self.machine.warehouse.inventory_1.demand.fulfilled == 12 and self.machine.warehouse.inventory_2.demand.fulfilled == 12:
            reward = -1

        elif self.machine.produce == 0:
            reward = 0
        elif self.machine.produce == 101:
            reward = 0
        elif self.machine.produce == 201:
            reward = -5

        #get observation
        #observation = self.machine.warehouse.current_warehouse_level[0], self.machine.warehouse.current_warehouse_level[1], self.machine.warehouse.demand.product[0], self.machine.warehouse.demand.product[1]
        observation = self.env._get_obs(self)

        #check if done
        done = False
        print(observation)
        return observation, reward, done, {}
    




    def _get_obs(self):
        """
        Returns the current observation.
        """
        #get observation for inventory
        p1_inventory = self.machine.warehouse.products[0].inventory_level
        p2_inventory = self.machine.warehouse.products[1].inventory_level

        #get observation for demand
        p1_demand = self.machine.warehouse.products[0].demand_class.demand[0]
        p2_demand = self.machine.warehouse.products[1].demand_class.demand[0]


        return p1_inventory, p2_inventory, p1_demand, p2_demand


    def reset(self):
        pass


    def render(self):
        pass


    def close(self):
        pass