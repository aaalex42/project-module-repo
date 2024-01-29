from collections import deque, namedtuple
import warnings
import random
import numpy as np

# This is a transition, given "state0", performing "action" yields "reward" and results in "state1"
# that might be "terminal"
EXPERIENCE = namedtuple('Experience', 'state0, action, reward, state1, terminal1')


def sample_batch_indices(low, high, size):
    """
    - Return a sample of (size) unique elements between low and high.
    - When there is enough data, draw without replacement. Otherwise, draw with replacement.
    - np.random.choice is inefficient as the memory grows. Therefore, random.sample is used 
        which is faster (for drawing without replacement).
    """
    if high - low >= size:
        # There is enough data to sample without replacement.
        #batch_idx = np.random.default_rng().choice(range(low, high), size=size, replace=False)
        batch_idx = random.sample(range(low, high), size)
    else:
        # There is not enough data to sample without replacement.
        warnings.warn("Not enough data to sample without replacement. Sampling with replacement. Or consider increasing the warm-up phase.")
        #batch_idx = np.random.default_rng().choice(range(low, high), size=size, replace=True)
        batch_idx = np.random.default_rng().integers(low, high - 1, size=size) #CHECK HEREEEEEEEEEEEEEEE why here it was high - 1, because it is an index and lists are up to high - 1

    assert len(batch_idx) == size
    return batch_idx


class RingBuffer(object):
    def __init__(self, maxlen) -> None:
        self.maxlen = maxlen
        self.start = 0
        self.length = 0
        self.data = [None] * self.maxlen
   
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        if idx < 0 or idx >= self.length:
            raise KeyError()
        return self.data[(self.start + idx) % self.maxlen]
    
    def append(self, v):
        if self.length < self.maxlen:
            # There is enough space, increase the length.
            self.length += 1
        elif self.length == self.maxlen:
            # No space, delete the first item
            self.start = (self.start + 1) % self.maxlen
        else:
            # This must not happen.
            raise RuntimeError()
        self.data[(self.start + self.length - 1) % self.maxlen] = v


class Memory(object):
    def __init__(self, window_length, ignore_episode_boundaries = False) -> None:
        self.window_length = window_length
        self.ignore_episode_boundaries = ignore_episode_boundaries

        self.recent_observations = deque(maxlen=window_length)
        self.recent_terminals = deque(maxlen=window_length)
      
    def sample(self, batch_size, batch_idx):
        raise NotImplementedError
    
    def append(self, observation, action, reward, terminal, training = True):
        """
        Append a new experience to the memory.
        """
        self.recent_observations.append(observation)
        self.recent_terminals.append(terminal)
  
    def get_recent_state(self, current_observation):
        """
        It could be the case when the subsequent observations are from different episodes.
        Therefore, it is necessary to ensure that an experience never spans over multiple episodes.
        """
        state = [current_observation]
        idx = len(self.recent_observations) - 1
        for offset in range(self.window_length - 1):
            current_idx = idx - offset
            current_terminal = self.recent_terminals[current_idx - 1] if current_idx - 1 >= 0 else False
            if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                # The previous observation was terminal. Do not add the current one.
                # Otherwise, it would leak into a different episode.
                break
            state.insert(0, self.recent_observations[current_idx])

        while len(state) < self.window_length:
            state.insert(0, np.zeros_like(current_observation))
        
        return state
      
    def get_config(self):
        return {
            "window_length": self.window_length,
            "ignore_episode_boundaries": self.ignore_episode_boundaries
        }
    

class SequentialMemory(Memory):
    def __init__(self, limit, **kwargs):
        super(SequentialMemory, self).__init__(**kwargs)

        self.limit = limit

        # deque is slow on random access. Thus, RingBuffer is used.
        self.actions = RingBuffer(limit)
        self.rewards = RingBuffer(limit)
        self.terminals = RingBuffer(limit)
        self.observations = RingBuffer(limit)

    def sample(self, batch_size, batch_idxs = None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indices(0, self.nb_entries - 1, size=batch_size)
        batch_idxs = np.array(batch_idxs) + 1
        assert np.min(batch_idxs) >= 1 
        assert np.max(batch_idxs) < self.nb_entries
        assert len(batch_idxs) == batch_size

        # Create experiences
        experiences = []
        for idx in batch_idxs:
            terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            while terminal0:
                # Skip the transition because the environment was reset here.
                # Select a new random transition and use this instead.
                # This can cause the batch ton contain the same transition twice.
                idx = sample_batch_indices(1, self.nb_entries, size = 1)[0]
                terminal0 = self.terminals[idx - 2] if idx >= 2 else False
            assert 1 <= idx < self.nb_entries

            # There could be a case when subsequent observations might be from different episodes..
            # Thus, it is ensured that an experience never spans over multiple episodes.
            # This is probably no that important in practice but it seems cleaner.
            state0 = [self.observations[idx - 1]]
            for offset in range(self.window_length - 1):
                current_idx = idx - 2 - offset
                current_terminal = self.terminals[current_idx - 1] if current_idx - 1 > 0 else False
                if current_idx < 0 or (not self.ignore_episode_boundaries and current_terminal):
                    # The previous observation was terminal. Do not add the current one.
                    # Otherwise, it would leak into a different episode.
                    break
                state0.insert(0, self.observations[current_idx])
            
            while len(state0) < self.window_length:
                state0.insert(0, np.zeros_like(state0[0]))
            action = self.actions[idx - 1]
            reward = self.rewards[idx - 1]
            terminal1 = self.terminals[idx - 1]


            # A follow-up state is needed. This means shifting state0 on timestep to the right.
            # It is important to be careful not to include an observation from the next episode 
            # if the last state is terminal.
            state1 = [np.copy(x) for x in state0[1:]]
            state1.append(self.observations[idx])

            assert len(state0) == self.window_length
            assert len(state1) == len(state0)
            experiences.append(EXPERIENCE(state0 = state0, action = action, reward = reward,
                                          state1 = state1, terminal1 = terminal1))
        
        assert len(experiences) == batch_size
        return experiences
    
    def sample_and_split(self, batch_size, batch_idxs = None):
        experiences = self.sample(batch_size, batch_idxs)

        # Split the batch into individual lists
        state0_batch = []
        action_batch = []
        reward_batch = []
        state1_batch = []
        terminal1_batch = []
        for exp in experiences:
            state0_batch.append(exp.state0)
            action_batch.append(exp.action)
            reward_batch.append(exp.reward)
            state1_batch.append(exp.state1)
            terminal1_batch.append(exp.terminal1)
        
        # Convert the lists to numpy arrays
        state0_batch = np.array(state0_batch).reshape(batch_size, -1)
        action_batch = np.array(action_batch).reshape(batch_size, -1)
        reward_batch = np.array(reward_batch).reshape(batch_size, -1)
        state1_batch = np.array(state1_batch).reshape(batch_size, -1)
        terminal1_batch = np.array(terminal1_batch).reshape(batch_size, -1)
        return state0_batch, action_batch, reward_batch, state1_batch, terminal1_batch
    
    def append(self, observation, action, reward, terminal, training = True):
        super(SequentialMemory, self).append(observation, action, reward, terminal, training)

        # In "observation", take "action", get "reward" and check if next state is "terminal"
        if training:
            self.observations.append(observation)
            self.actions.append(action)
            self.rewards.append(reward)
            self.terminals.append(terminal)
    
    @property
    def nb_entries(self):
        return len(self.observations)
    
    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config["limit"] = self.limit
        return config


class EpisodeParameterMemory(Memory):
    def __init__(self, limit, **kwargs) -> None:
        super(EpisodeParameterMemory, self).__init__(**kwargs)
        
        self.limit = limit
        self.params = RingBuffer(limit)
        self.intermediate_rewards = []
        self.total_rewards = RingBuffer(limit)
    
    def sample(self, batch_size, batch_idxs = None):
        if batch_idxs is None:
            batch_idxs = sample_batch_indices(0, self.nb_entries, size = batch_size)
        assert len(batch_idxs) == batch_size

        batch_params = []
        batch_total_rewards = []
        for idx in batch_idxs:
            batch_params.append(self.params[idx])
            batch_total_rewards.append(self.total_rewards[idx])
    
    def append(self, observation, action, reward, terminal, training = True):
        super(EpisodeParameterMemory, self).append(observation, action, reward, terminal, training)
        if training:
            self.intermediate_rewards.append(reward)
    
    def finalize_episode(self, params):
        total_reward = sum(self.intermediate_rewards)
        self.total_rewards.append(total_reward)
        self.params.append(params)
        self.intermediate_rewards = []
    
    @property
    def nb_entries(self):
        return len(self.total_rewards)
    
    def get_config(self):
        config = super(SequentialMemory, self).get_config()
        config["limit"] = self.limit
        return config