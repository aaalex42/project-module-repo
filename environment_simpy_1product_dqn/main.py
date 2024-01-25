#from gym_environment import InventoryManagementEnv
from psim_gym import InventoryManagementEnv
from q_learning_agent import QLearningAgent
from dqn_agent import DQNAgent

def main():
    # AR changed to empty parameter list, set by default
    env = InventoryManagementEnv()
    #print(env.dl[0].demands)    
    agent = DQNAgent(env, device='cuda', num_episodes=250, BATCH_SIZE=10000, GAMMA=0.99, EPS_START=0.9, EPS_END=0.05, EPS_DECAY=1000, TAU=0.005, LR=1e-5)
    total_rewards, inventory_levels, _ = agent.train() #if num_episodes increases, simulation time has to be increased as well in psim_gym.py
    agent.plot_rewards(total_rewards, inventory_levels)

    env.post_sim_analysis()

if __name__ == "__main__":
    main()