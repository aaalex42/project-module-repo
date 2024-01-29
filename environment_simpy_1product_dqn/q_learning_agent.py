import numpy as np
import matplotlib.pyplot as plt

# AR added this
VERBOSE = False

class QLearningAgent:
    def __init__(self, env, alpha=0.1, gamma=0.6, epsilon=0.4, num_bins=100):
        self.env = env
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon
        self.num_bins = num_bins
        self.q_table = np.zeros((env.max_inventory, env.max_demand, env.max_action))
        self.inventory_bins = np.linspace(0, env.max_inventory, num_bins, dtype=np.int32)
        self.demand_bins = np.linspace(0, env.max_demand, num_bins, dtype=np.int32)

        self.total_rewards = []
        self.inventory_levels = []

    def discretize_state(self, obs): # Convert the states to a discrete state
        return (np.digitize(obs[0], self.inventory_bins) - 1, np.digitize(obs[1], self.demand_bins) - 1)

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

    def learn(self, episode):        
        total_reward = 0
        obs = self.env.reset()
        for episode in range(episode):
            state = self.discretize_state(obs)
            action = self.choose_action(state)
            next_obs, reward, done, _ = self.env.step(action)
            total_reward += reward
            next_state = self.discretize_state(next_obs)
            self.update_q_table(state, action, reward, next_state)
            obs = next_obs
            self.total_rewards.append(total_reward)
            self.inventory_levels.append(obs[0])
            if done:                
                total_reward = 0
                obs = self.env.reset()
                break
            if VERBOSE:
                print(f"Step {episode + 1}:")
                print(f"Action Taken (order_amount): {action}")
                print(f'Environment return following States:')
                print(f"Inventory Level: {obs[0]}, Demand_open: {self.env.demand_open}, New_Demand: {self.env.new_demand}")
            
        print(f"Total Reward after {episode + 1} Epsiodes: {total_reward}")
        print("-" * 40)
        return self.total_rewards, self.inventory_levels
    
    def plot_rewards(self, total_rewards, inventory_levels):
        fig, ax1 = plt.subplots()

        color1 = 'tab:blue'
        color2 = 'tab:green'

        ax1.set_xlabel('Episode')
        ax1.set_ylabel('Inventory Level', color=color1)
        ax1.plot(inventory_levels, color=color1)
        ax1.tick_params(axis='y', labelcolor=color1)

        ax2 = ax1.twinx()  # instantiate a second axes that shares the same x-axis
        ax2.set_ylabel('Total Reward', color=color2)
        ax2.plot(total_rewards, color=color2)
        ax2.tick_params(axis='y', labelcolor=color2)

        fig.tight_layout()  # otherwise the right y-label is slightly clipped
        plt.show()