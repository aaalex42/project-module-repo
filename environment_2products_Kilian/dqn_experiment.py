from environment_2products_Kilian.dqn_environment import Production_DQN_Env
from environment_2products_Kilian.dqn_agent import DQNAgent



env = Production_DQN_Env()
agent = DQNAgent(env.observation_space, env.action_space)

order = ([1000,2000])

env.step(order)
