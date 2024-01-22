#from gym_environment import InventoryManagementEnv
from psim_gym import InventoryManagementEnv
from q_learning_agent import QLearningAgent

# AR changed to empty parameter list, set by default
env = InventoryManagementEnv()
agent = QLearningAgent(env, alpha=0.1, gamma=0.6, epsilon=0.4)

agent.learn(episodes=20)
agent.plot_rewards()
env.post_sim_analysis()

