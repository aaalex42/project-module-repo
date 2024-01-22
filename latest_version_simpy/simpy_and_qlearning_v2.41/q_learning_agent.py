import numpy as np


# AR added this
VERBOSE = False

"""
AR removed all references to bins, this is taken care of in env
AR switched from episode length to number of episodes. each episode is ended by done flag from env
"""


class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.6, epsilon=0.4, num_bins=100):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.q_table = np.zeros((env.max_inventory, env.max_demand, env.max_action))

        self.total_rewards = []
        
    def choose_action(self, state): # Choose action: either explore or exploit
        if np.random.uniform(0, 1) < self.epsilon:
            return self.env.action_space.sample()  # Explore
        else:
            return np.argmax(self.q_table[state])  # Exploit

    def update_q_table(self, state, action, reward, next_state): 
        old_value = self.q_table[state + (action,)]
        next_max = np.max(self.q_table[next_state])
        new_value = (1 - self.alpha) * old_value + self.alpha * (reward + self.gamma * next_max)
        self.q_table[state + (action,)] = new_value

    def learn(self, episodes):        
        for episode in range(episodes):
            total_reward = 0
            obs = self.env.reset()
            done = False
            i = 0
            while not done:
                i += 1
                state = tuple(obs)
                action = self.choose_action(state)
                next_obs, reward, done, _ = self.env.step(action)
                total_reward += reward
                next_state = tuple(next_obs)
                self.update_q_table(state, action, reward, next_state)
                obs = next_obs
                self.total_rewards.append(total_reward)
                if VERBOSE:
                    print(f"Step {episode + 1}:")
                    print(f"Action Taken (order_amount): {action}")
                    print(f'Environment return following States:')
                    print(f"Inventory Level: {obs[0]}, Demand_open: {self.env.demand_open}, New_Demand: {self.env.new_demand}")
                
            if True:
                print(f"Total relative reward after {episode + 1} epsiodes: {total_reward / i}")
                print("-" * 40)
    
    def plot_rewards(self):
        import matplotlib.pyplot as plt
        plt.plot(self.total_rewards)
        plt.xlabel('Step')
        plt.ylabel('Total Reward')
        plt.show()  
