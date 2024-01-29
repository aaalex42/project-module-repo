import gym 
from gym import spaces
from init_vars import *
from classes_n import *
import numpy as np
from gym.core import Env


class Production_DQN_Env(gym.Env):
    """
    This class is a wrapper for the production classes according to the OpenAI Gym rules.
    It is based on discrete action and observation spaces.

    ### Action space:
        - id0: 0 or 1: product 1 or 2
        - id1: 0 to SOME_QUANTITY: quantity to produce
    
        | Num | Action   | Min   | Max                               | Type     |
        |-----|----------|-------|-----------------------------------|----------|
        | 0   | Product  | 0     | 1                                 | Discrete |
        | 1   | Quantity | 0     | MAXIMUM_INVENTORY // BIN_SIZE + 1 | Discrete |

    ### Observation space:
        The observation is a `ndarray` with shape `(5,)`.

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
        | 4   | Machine is busy  | 0 (F)| 1 (True)              | Discrete |

    ### Reward:
        The reward function is defined as:
        R = 1 - current_warehouse_level / MAXIMUM_INVENTORY
    
    ### Step:
        One step is defined as an hour of a day out of HOURS_PER_DAY hours (usually 24).

    ### Starting state:

    ### Episode truncation:

    ### Arguments:
        All arguments are given in init_vars.py file.

    ### Version history:
        No version history yet.
    """
    def __init__(self, verbose = 0):
        self.action_space = spaces.MultiDiscrete([2, MAXIMUM_INVENTORY // BIN_SIZE + 1])
        self.observation_space = spaces.MultiDiscrete([MAXIMUM_INVENTORY, MAXIMUM_INVENTORY, MAXIMUM_INVENTORY / 2, MAXIMUM_INVENTORY / 2, 2])

        #initialize the demands, inventories, warehouse and machine
        self.machine = Machine(
            Warehouse(
                Inventory(Demand(P1)),
                Inventory(Demand(P2))
            )
        )

        #for displaying the information
        self.verbose = verbose

        self.step_count = 0


    def step(self, action): 
        #action_np = action.cpu().numpy() #convert tensor to numpy array because assert statement does not work with tensors      
        
        #print(action)
        #check if action is in the action space Multidiscrete [(product id and quantity)]
        #print(self.action_space)
        #print(action[0].item(), action[1].item())
        assert self.action_space.contains([action[0].item(), action[1].item()])
        
        
        
        self.step_count += 1
        
        
        if self.step_count >= 5000: #5000 days
            truncated = True
        else:
            truncated = False

        terminated = False

        #check if maximum warehouse level is exceeded, or inventory is negative
        #for both cases, the game ends with negative reward
        """CHECK IF IT IS EVEN POSSIBLE TO HAVE NEGATIVE INVENTORY OR EXCEED MAXIMUM WAREHOUSE LEVEL"""
        if (self.machine.warehouse.current_warehouse_level > MAXIMUM_INVENTORY   
                or self.machine.warehouse.products[0].inventory_level[-1] < 0  
                or self.machine.warehouse.products[1].inventory_level[-1] < 0):
            #give negative reward and restart the game
            reward = -1
            terminated = True

            return self._get_obs(), reward, terminated, False, {}

        exit_code_prod = self.machine.produce((action[0].item(), action[1].item() * BIN_SIZE)) #exit code not used, because the main reward is coming from fulfillment
        exit_code_fulf = self.machine.fulfill()
        exit_code_stor = self.machine.store_production()

        if self.verbose == 1:
            print("    exit code prod: ", exit_code_prod)
            print("    exit code fulf (p1, p2): ", exit_code_fulf)
            print("    exit code stor: ", exit_code_stor)
            print("    service level p1: ", self.machine.warehouse.products[0].demand_class.service_level)
            print("    service level p2: ", self.machine.warehouse.products[1].demand_class.service_level)

        reward = 0
        #if exit_code_fulf[0] == 10:
            # not the right hour to fulfill an order
            #reward += 0
            #print("    -reward: ", reward)
        #if exit_code_fulf[0] == 100 or exit_code_fulf[1] == 100:
            # no demand in that day, reward 0
            #reward += 0
            #print("    -reward exit code 100: ", reward)
        """IMPLEMENT MAYBE TO SUM UP THE REWARDS FROM 2 PRODUCTS IF BOTH ARE FULFILLED"""
        if exit_code_fulf[0] == 101 or exit_code_fulf[1] == 101:
            # the demand has been fulfilled 
            reward += 1 - self.machine.warehouse.current_warehouse_level / MAXIMUM_INVENTORY
        if exit_code_fulf[0] == 102 or exit_code_fulf[1] == 102:
            # the demand has not been fulfilled
            reward += -1

        if ((reward != 0) and 
                (self.machine.warehouse.products[0].demand_class.service_level < SERV_LVL_P1
                or self.machine.warehouse.products[1].demand_class.service_level < SERV_LVL_P2)):
            # the service level is not met
            reward = -1
            terminated = True
        
        print("step: ", self.step_count)
        print(f' observation {self._get_obs()},    reward: {reward:.3f}, terminated: {terminated}, truncated: {truncated}')
        #return observation, reward, terminated, False, {}
        
        return self._get_obs(), reward, terminated, truncated, False, {}


    def _get_obs(self):
        return (self.machine.warehouse.products[0].inventory_level[-1],
                self.machine.warehouse.products[1].inventory_level[-1], 
                self.machine.warehouse.products[0].demand_class.demand[self.machine.t // HOURS_PER_DAY],
                self.machine.warehouse.products[1].demand_class.demand[self.machine.t // HOURS_PER_DAY],
                self.machine.check_if_machine_free()
                #self.machine.t//24, self.machine.t%24
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

    
    def inc_t(self):
        self.machine.t += 1

class SkipStep(gym.Wrapper):
    def __init__(self, env: Env, skip: int):
        super().__init__(env)
        self._skip = skip
    
    def step(self, action):
        """Repeat an action and sum the reward"""
        total_reward = 0.0
        for i in range(self._skip):
            #Accumulate the reward and repeat the same action
            obs, reward, done, truncated, _, _  = self.env.step(action)
            total_reward += reward
            self.env.inc_t()
            if done:
                break
        return obs, total_reward, done, truncated, {}, {}