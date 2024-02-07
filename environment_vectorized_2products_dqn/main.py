from dqn_bin_environment import *
from dqn_agent import *


def main():
    env = Production_DQN_Env()

    skip_env = SkipStep(env, skip=24)

    #print(env.action_space.shape[0])

    agent = DQNAgent(skip_env, device='cuda', num_episodes=1000, BATCH_SIZE=512, BATCH_START_SIZE=10000, MEMORY_SIZE=100000, GAMMA=0.99, EPS_START=0.9, EPS_END=0.02, EPS_DECAY=5000, TAU=0.005, LR=1e-5)
    total_rewards, inventory_levels, _ = agent.train() #if num_episodes increases, simulation time has to be increased as well in psim_gym.py
    agent.plot_rewards(total_rewards, inventory_levels)
    agent.plot_durations(show_result=True)
    plt.ioff()
    plt.show()


if __name__ == "__main__":
    main()
