from dqn_environment import Production_DQN_Env
from dqn_agent import DQNAgent



env = Production_DQN_Env()
agent = DQNAgent(env.observation_space, env.action_space)

# order [order_amount_product1, order_amount_product2]
action = [1000,2000]

env.step(action)

