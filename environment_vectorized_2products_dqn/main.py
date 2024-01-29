from dqn_bin_environment import *
from dqn_agent import *



env = Production_DQN_Env()
#print(env.action_space.shape[0])

agent = DQNAgent(env, device='cuda', num_episodes=10, BATCH_SIZE=128, GAMMA=0.99, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=1000, TAU=0.005, LR=1e-5)
total_rewards, inventory_levels, _ = agent.train() #if num_episodes increases, simulation time has to be increased as well in psim_gym.py
agent.plot_rewards(total_rewards, inventory_levels)