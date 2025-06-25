import math
import os
import itertools
import time
from collections import namedtuple, deque

import numpy as np
import torch
from torch import optim, Tensor, nn
from matplotlib import pyplot as plt
from torch.backends.cudnn import deterministic

from rl_power.envs.branch_env import BranchEnv
from rl_power.envs.rllib_multi_branch_env import RLLibBranchEnv
from rl_power.envs.sampled_branch_env import SampledBranchEnv
from rl_power.modules.branch_policy_model import DefaultNetwork
from rl_power.training.power_eval import run_policy_on_branch_env
from rl_power.training.replay_buffer import LSTMTransition, Transition, ReplayMemory
from rl_power.visualization.visualization import plot_training_curve
import random
from typing import Dict, Any, Union
from itertools import count


class BranchEnvTrainer:
    def __init__(self, env_sampling_config: dict, use_lstm: bool = False, model_linear_dim: int = 256, max_actions: Union[int, list] = 10, n_agents: int = 2, batch_size: int = 128):
        self.env_sampling_config = env_sampling_config
        self.use_lstm = use_lstm
        self.max_actions = max_actions
        self.n_agents = n_agents
        self.discount_rate = 0.99
        self.buffer_size = 10000
        self.batch_size = batch_size
        self.eps_start = 0.99
        self.eps_end = 0.05
        self.eps_decay = 1000
        self.tau = 0.005
        self.lr = 1e-4
        self.steps_done = 0
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.criterion = nn.SmoothL1Loss()

        self.env = SampledBranchEnv(env_sampling_config, max_actions=max_actions, n_agents=n_agents)

        # Get number of actions from gym action space
        self.n_actions = self.env.action_space.n

        # Get the number of state observations
        state, info = self.env.reset()
        self.n_observations = len(state[list(state.keys())[0]])

        self.policy_net = DefaultNetwork(state_length=self.n_observations, n_actions=self.n_actions, use_lstm=use_lstm, linear_dim=model_linear_dim)
        self.target_net = DefaultNetwork(state_length=self.n_observations, n_actions=self.n_actions, use_lstm=use_lstm, linear_dim=model_linear_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())

        self.optimizer = optim.AdamW(self.policy_net.parameters(), lr=self.lr, amsgrad=True)
        self.memory = ReplayMemory(self.buffer_size, use_lstm=use_lstm)

        self.reward_history = []
        self.episode_length_history = []

    def select_action(self, state: Dict[str, Tensor], hidden_state=None, deterministic=False):
        sample = random.random()
        eps_threshold = self.eps_end + (self.eps_start - self.eps_end) * math.exp(
            -1. * self.steps_done / self.eps_decay)
        self.steps_done += 1

        tensor_state = torch.cat([v for v in list(state.values())], dim=0)

        action_vector = None

        branches = list(state.keys())

        if sample > eps_threshold or deterministic:
            with torch.no_grad():
                if self.use_lstm:
                    policy_output, next_hidden_state = self.policy_net(tensor_state, hidden_state)
                    policy_output = policy_output.max(1).indices.view(-1, 1)
                else:
                    policy_output, next_hidden_state = self.policy_net(tensor_state, hidden_state)
                    policy_output = policy_output.max(1).indices.view(-1, 1)
                    next_hidden_state = None

                action_vector = {b: int(policy_output[i].item()) for i, b in enumerate(branches)}
        else:
            sampled_output = torch.tensor([[self.env.action_space.sample() for _ in state.keys()]], device=self.device,
                                          dtype=torch.long)
            action_vector = {b: int(sampled_output[0, i].item()) for i, b in enumerate(branches)}
            policy_output, next_hidden_state = self.policy_net(tensor_state, hidden_state)

        return action_vector, next_hidden_state

    def train(self, n_iterations: int = 100):

        for i_episode in range(n_iterations):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = {branch: torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                     for branch, obs in state.items()}

            if i_episode % 100 == 0:
                print("Average reward (last 100 episodes):" + str(np.mean(self.reward_history[-100:])))

            self.episode_length_history.append(0)
            self.reward_history.append(0)

            hidden_state = None

            for t in count():
                action, hidden_state = self.select_action(state, hidden_state)
                observation, reward, terminated, truncated, _ = self.env.step(action)

                terminated = terminated["__all__"]
                truncated = terminated

                reward = {branch: torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(0)
                          for branch, r in reward.items()}

                # self.reward_history[i_episode].append(np.mean([v.detach().cpu() for v in list(reward.values())]))
                self.episode_length_history[i_episode] += 1

                done = terminated or truncated

                self.reward_history[i_episode] += np.mean([v.detach().cpu() for v in list(reward.values())])

                if terminated:
                    next_state = {branch: None for branch, obs in observation.items()}
                    # next_state = None
                    # next_state = {branch: torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                    #               for branch, obs in observation.items()}
                else:
                    next_state = {branch: torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                                  for branch, obs in observation.items()}

                # Store the transition in memory
                if self.use_lstm:

                    hidden_state_h = {branch: hidden_state[0][:, i, :].detach()
                                      for i, branch in enumerate(observation.keys())}

                    hidden_state_c = {branch: hidden_state[1][:, i, :].detach()
                                      for i, branch in enumerate(observation.keys())}

                    hidden_state_store = (hidden_state_h, hidden_state_c)

                    self.memory.push(state, action, next_state, reward, hidden_state_store, device=self.device)
                else:
                    self.memory.push(state, action, next_state, reward, device=self.device)

                state = next_state
                self.update_model()

                # Soft update of the target network's weights
                # θ′ ← τ θ + (1 −τ )θ′
                target_net_state_dict = self.target_net.state_dict()
                policy_net_state_dict = self.policy_net.state_dict()
                for key in policy_net_state_dict:
                    target_net_state_dict[key] = policy_net_state_dict[key] * self.tau + target_net_state_dict[key] * (
                            1 - self.tau)
                self.target_net.load_state_dict(target_net_state_dict)

                if done:
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    break

        time_string = time.strftime("%Y%m%d%H%M%S")
        if not os.path.exists("./pytorch_models/"):
            os.makedirs("./pytorch_models/")
        torch.save(self.policy_net.state_dict(), f"./pytorch_models/dqn_policy_weights_{time_string}.pth")

    def update_model(self):

        if len(self.memory) < self.batch_size:
            return

        transitions = self.memory.sample(self.batch_size)

        if self.use_lstm:
            batch = LSTMTransition(*zip(*transitions))
        else:
            batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=self.device, dtype=torch.bool)

        non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        if self.use_lstm:
            hidden_batch_h = torch.cat(batch.lstm_state)
            hidden_batch_c = torch.cat(batch.lstm_cell)
            hidden_batch_h = hidden_batch_h.view(1, *hidden_batch_h.shape)
            hidden_batch_c = hidden_batch_c.view(1, *hidden_batch_c.shape)

            hidden_batch = (hidden_batch_h, hidden_batch_c)

            # Mask non-final states.
            hidden_batch_h = hidden_batch_h[:, non_final_mask, :]
            hidden_batch_c = hidden_batch_c[:, non_final_mask, :]
            non_final_hidden_batch = (hidden_batch_h, hidden_batch_c)

        else:
            hidden_batch = None
            non_final_hidden_batch = None

        if self.use_lstm:
            state_action_values, hiddens = self.policy_net(state_batch, hidden_batch)
            state_action_values = state_action_values.gather(1, action_batch.reshape(-1, 1))
        else:
            state_action_values, hiddens = self.policy_net(state_batch)
            state_action_values = state_action_values.gather(1, action_batch.reshape(-1, 1))

        next_state_values = torch.zeros(self.batch_size, device=self.device)
        with torch.no_grad():
            _next_values, _next_hidden = self.target_net(non_final_next_states, non_final_hidden_batch)
            next_state_values[non_final_mask] = _next_values.max(1).values
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.discount_rate) + reward_batch

        # Compute Huber loss
        loss = self.criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        # In-place gradient clipping
        torch.nn.utils.clip_grad_value_(self.policy_net.parameters(), 100)
        self.optimizer.step()

    def evaluate(self, path: str, active_branches: list[str] = None, max_actions: int = 5):
        config = {"render_mode": True, "path": path, "max_actions": max_actions, "n_agents": len(active_branches)}
        test_env = RLLibBranchEnv(config)
        state, info = test_env.reset()

        if active_branches is not None:
            test_env.active_branches = active_branches

        state = test_env.get_observation()

        terminated = False
        hidden_state = None
        while not terminated:
            state = {branch: torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                     for branch, obs in state.items()}

            action, hidden_state = self.select_action(state, hidden_state, deterministic=True)
            state, reward, terminated, truncated, _ = test_env.step(action)
            print(f"action: {action} \nreward: {reward}\n\n")
            terminated = terminated["__all__"]

    def full_evaluation(self, path: str, n_agents: int = 3, max_actions: int = 5, limit: int = None, plot: bool = False):

        trial_rewards = []

        config = {"render_mode": False, "path": path, "max_actions": max_actions, "n_agents": n_agents}
        test_env = RLLibBranchEnv(config)

        counter = 0

        for active_branch_list in itertools.combinations(test_env.branches, n_agents):

            if limit is not None:
                counter += 1
                if counter >= limit:
                    break

            state, info = test_env.reset()
            test_env.active_branches = active_branch_list

            state = test_env.get_observation()
            terminated = False
            hidden_state = None

            while not terminated:
                state = {branch: torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                         for branch, obs in state.items()}

                action, hidden_state = self.select_action(state, hidden_state, deterministic=True)
                state, reward, terminated, truncated, _ = test_env.step(action)
                print(f"action: {action} \nreward: {reward}\n\n")
                terminated = terminated["__all__"]

            trial_rewards.append(np.mean(list(reward.values())))

        if plot:
            plt.figure()
            plt.hist(trial_rewards, bins=20)
            plt.figure()
            plt.plot(trial_rewards, np.zeros(len(trial_rewards)), 'x')
            print(f"Mean trial reward: {np.mean(trial_rewards)}\nMax trial reward: {max(trial_rewards)}\nMin trial reward: {min(trial_rewards)}")
            plt.show()

        return trial_rewards

    def agent_count_evaluation_sweep(self, test_case: str, min_agents: int = 1, max_agents: int = 5):

        n_plots = max_agents - min_agents + 1
        fig_bar, axs_bar = plt.subplots(n_plots, 1, sharex=True)
        fig_scatter, axs_scatter = plt.subplots(n_plots, 1, sharex=True)

        max_n_agent = max_agents

        for i, agent_count in enumerate(list(range(min_agents, max_n_agent + 1))):

            r = self.full_evaluation(os.path.abspath(test_case), n_agents=agent_count, plot=False)
            mean_val = np.mean(r)

            # Histogram plot.
            axs_bar[i].hist(r, density=True)
            axs_bar[i].scatter(mean_val, 0, marker='x', color='r')

            # Error bar plot.

            min_gap = abs(mean_val - min(r))
            max_gap = abs(mean_val - max(r))
            axs_scatter[i].errorbar(mean_val, 0, xerr=[[min_gap], [max_gap]], fmt="o")

        fig_bar.suptitle(f"Reward histograms: n_agents = {min_agents}-{max_agents} | {test_case}")

    def plot_history(self):

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.reward_history)
        axs[1].plot(self.episode_length_history)
        plt.show()

    def load_latest_model(self):

        path = "./pytorch_models/"

        files = os.listdir(path)
        paths = [os.path.join(path, basename) for basename in files]
        latest_model = max(paths, key=os.path.getctime)

        self.policy_net.load_state_dict(torch.load(latest_model, weights_only=True))


if __name__ == '__main__':
    # sampler_options = {"paths": [os.path.abspath("ieee_data/WB5.m"),
    #                              os.path.abspath("ieee_data/pglib_opf_case14_ieee.m"),
    #                              os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
    #                              os.path.abspath("ieee_data/pglib_opf_case57_ieee.m")],
    #                    "weights": [0.1, 0.4, 0.3, 0.2]
    #                    }

    sampler_options = {"paths": [os.path.abspath("ieee_data/WB5.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case14_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case57_ieee.m")],
                       # "weights": [1, 2, 4, 8]
                       "weights": [1, 2, 4, 8]
                       }
    use_lstm = False
    # load_from_memory = True
    load_from_memory = False

    print(f"LSTM = {use_lstm}")
    trainer = BranchEnvTrainer(sampler_options, use_lstm=use_lstm, model_linear_dim=256, max_actions=[5, 15], n_agents=5, batch_size=512)

    if load_from_memory:
        trainer.load_latest_model()
        results = None
    else:
        results = trainer.train(10000)

    test_case = "ieee_data/WB5.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=5)

    test_case = "ieee_data/pglib_opf_case14_ieee.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=4)

    test_case = "ieee_data/pglib_opf_case30_ieee.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=3)

    test_case = "ieee_data/pglib_opf_case57_ieee.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=2)

    # trainer.full_evaluation(), n_agents=2)

    trainer.plot_history()
    trainer.evaluate(os.path.abspath("ieee_data/pglib_opf_case57_ieee.m"), active_branches=['8'])


    print("")

    # run_policy_on_branch_env(eval_env, algo, 5)
