import torch


class Memory:
    def __init__(self, capacity: int, gamma: float = 1.0) -> None:
        self.capacity = capacity
        self.episode_reward_batch = []
        self.state_batch = []
        self.action_batch = []
        self.value_batch = []
        self.next_value_batch = []
        self.probs_batch = []
        self.log_probs_batch = []
        self.current_buffer_size = 0
        self.current_episode_start_index = 0
        self.gamma = gamma

    def save(self, state, action, value, next_value, log_probs, probs, episode_reward, terminated) -> bool:

        self.state_batch.append(state)
        self.action_batch.append(action)
        self.value_batch.append(value)
        self.next_value_batch.append(next_value)
        self.log_probs_batch.append(log_probs)
        self.probs_batch.append(probs)
        self.episode_reward_batch.append(episode_reward)

        self.current_buffer_size += 1

        if terminated:
            self.backfill_episode_returns()
            self.current_episode_start_index = self.current_buffer_size

        return len(self.state_batch) == self.capacity

    def load(self):
        states = torch.stack(self.state_batch)
        actions = torch.stack(self.action_batch)
        values = torch.cat(self.value_batch)
        next_values = torch.cat(self.next_value_batch)
        log_probs = torch.cat(self.log_probs_batch)
        probs = torch.stack(self.probs_batch)
        episode_reward = torch.cat(self.episode_reward_batch)
        episode_reward = (episode_reward - episode_reward.mean())/(episode_reward.std() + 1e-8)
        advantage = episode_reward.view((-1, 1)) - values.view((episode_reward.shape[0], -1)).detach()

        return states, actions, values, next_values, log_probs, probs, advantage, episode_reward

    def reset(self):
        self.state_batch = []
        self.value_batch = []
        self.next_value_batch = []
        self.log_probs_batch = []
        self.probs_batch = []
        self.episode_reward_batch = []
        self.action_batch = []
        self.current_buffer_size = 0
        self.current_episode_start_index = 0

    def backfill_episode_returns(self):
        start_idx = self.current_episode_start_index
        end_idx = self.current_buffer_size

        for i in range(end_idx - 2, start_idx - 1, -1):
            self.episode_reward_batch[i] += self.gamma * self.episode_reward_batch[i + 1]
