import copy
import itertools
import math
import os
import time
from collections import defaultdict
from itertools import count
from random import random
from typing import Tuple, Union, Dict
import dill as pickle
import uuid

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import tensor, nn, optim, Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from rl_power.envs.branch_env import BranchEnv
from rl_power.envs.edge_agent_branch_env import EdgeAgentBranchEnv
from rl_power.envs.node_agent_branch_env import NodeEnvSampler, SampledNodeEnv, NodeAgentBranchEnv
from rl_power.envs.rllib_multi_branch_env import RLLibBranchEnv
from rl_power.envs.sampled_branch_env import SampledBranchEnv
from rl_power.modules.bus_attention_model import BusAttentionActor, BusAttentionCritic
from rl_power.power.graph_utils import get_adjacent_branches
from rl_power.training.memory import Memory


class A2CBranchTrainer:
    def __init__(self, actor_type, critic_type, env_sampling_config: dict, model_linear_dim: int = 256,
                 model_attn_dim: int = 16,
                 max_actions: Union[int, list] = 10, n_agents: int = 2, device: str = "cpu", batch_size: int = 256,
                 entropy_coeff: float = 0.05, n_heads: int = 4, lr: float = 1e-3, n_actions: int = 3,
                 training_env=SampledNodeEnv):

        self.env_sampling_config = env_sampling_config
        self.max_actions = max_actions
        self.n_agents = n_agents
        self.discount_rate = 0.99
        self.lr = lr
        self.steps_done = 0
        self.device = device
        self.n_actions = n_actions
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff
        self.model_linear_dim = model_linear_dim
        self.env = training_env(env_sampling_config, max_actions=max_actions, n_agents=n_agents)

        self.memory = Memory(capacity=self.batch_size)

        # Get the number of state observations
        state, info = self.env.reset()
        first_agent = list(state.keys())[0]

        if isinstance(self.env, SampledNodeEnv):
            first_branch = list(state[first_agent].keys())[0]
            self.n_observations = len(state[first_agent][first_branch])
        else:
            first_branch = list(state.keys())[0]
            self.n_observations = len(state[first_branch])

        self.actor = actor_type(state_length=self.n_observations, n_actions=self.n_actions,
                                n_agents=n_agents,
                                attention_embed_dim=model_attn_dim,
                                linear_dim=model_linear_dim, device=self.device, n_heads=n_heads)
        self.critic = critic_type(state_length=self.n_observations, linear_dim=model_linear_dim,
                                  n_agents=n_agents,
                                  attention_embed_dim=model_attn_dim,
                                  device=self.device, n_heads=n_heads)
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr, betas=(0.1, 0.3))
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr, betas=(0.1, 0.3))

        self.reward_history = []
        self.episode_length_history = []
        self.critic_loss_history = []
        self.actor_loss_history = []
        self.actor_weight_grad_norm_history = []
        self.actor_weight_norm_history = []
        self.ppo_update_iterations = 4
        self.clip_ratio = 0.1
        time_string = time.strftime("%Y%m%d%H%M%S") + uuid.uuid4().hex
        self.save_directory = f"./results/{self.actor.__class__.__name__}_{time_string}/"


    def select_action(self, state: Dict[str, Tensor]):
        self.steps_done += 1

        # Execute actor/policy for all agents at once.
        stacked_states = torch.cat([b_state for b_state in state.values()])
        stacked_states = stacked_states.view(-1, *stacked_states.shape)
        stacked_action_probs = self.actor(stacked_states)

        dist = Categorical(stacked_action_probs)
        sampled_action = dist.sample().view(-1)
        # action_dict = {branch: sampled_action[i] for i, branch in enumerate(state.keys())}

        # action_vector = {b: int(policy_output[i].item()) for i, b in enumerate(branches)}

        return stacked_action_probs, sampled_action

    def train(self, n_iterations: int = 100):

        # start = time.time()

        for i_episode in range(n_iterations):
            # Initialize the environment and get its state
            state, info = self.env.reset()

            state = self.state_to_tensor(state, self.env)

            if i_episode % 100 == 0:
                print(f"Average reward as of episode {i_episode} (last 100 episodes):" + str(
                    np.mean(self.reward_history[-100:])))
                # end = time.time()
                # print(f"Elapsed time: {end-start}")
                # start = end

            self.episode_length_history.append(0)
            self.reward_history.append(0)

            for t in count():

                distribution, action = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action)

                # Convert next state from ndarray dict to tensor dict.
                next_state = self.state_to_tensor(next_state, self.env)

                truncated = terminated
                state_tensor = torch.cat(list(state.values()), dim=0)
                next_state_tensor = torch.cat(list(next_state.values()), dim=0)

                state_tensor = state_tensor.view(1, *state_tensor.shape)
                next_state_tensor = next_state_tensor.view(1, *next_state_tensor.shape)

                reward_list = [torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(0)
                               for branch, r in reward.items()]
                reward_tensor = torch.tensor(reward_list, device=self.device).mean()

                # if t == 0:
                #     episode_return = reward_tensor
                # else:
                #     episode_return += self.discount_rate * reward_tensor

                value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor)
                # td_target = reward_tensor + self.discount_rate * next_value.view(-1, 1)

                # if t >= self.max_actions - 1:
                #     td_target = reward_tensor + self.discount_rate * next_value.view(-1, 1)
                # else:
                #     td_target = reward_tensor + self.discount_rate * next_value.view(-1, 1) * (1 - terminated)
                # # td_target = (td_target - td_target.mean()) / (td_target.std() + 1e-8)

                self.memory.save(state_tensor, action, value, next_value, distribution,
                                 reward_tensor.view(-1), terminated)

                if terminated and (i_episode + 1) % self.batch_size == 0:
                    _states, _actions, _values, _next_values, _probs, _advantages, _episode_returns = self.memory.load()

                    # self.critic_update(_values, _episode_returns)
                    # self.actor_update(_probs, _log_probs, _advantages)
                    # self.memory.reset()
                    old_probs = _probs
                    for ppo_update_idx in range(self.ppo_update_iterations):
                        self.critic_update(_values, _episode_returns)
                        _values = self.critic(_states)
                        # _advantages = (_episode_returns.view(-1, 1) - _values).detach()
                        self.actor_update(_probs, _advantages, _actions, old_probs)
                        _probs = self.actor(_states).view(*_probs.shape)

                    self.memory.reset()


                self.episode_length_history[i_episode] += 1
                self.reward_history[i_episode] += np.mean(list(reward.values()))
                # print(np.mean(list(reward.values())))
                done = terminated or truncated
                state = next_state
                if done:
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    break

        self.save_model()

    def actor_update(self, probs, advantages, actions, old_probs=None):

        # stacked_probs = torch.cat(self.probs_batch, dim=0)
        # entropy = torch.sum(stacked_probs * stacked_probs.log(), dim=-1).mean()
        # entropy = torch.sum(probs * probs.log(), dim=-1).mean()
        # entropy = torch.nan_to_num(entropy, posinf=0, neginf=0)

        # adv_loss = torch.mean(-log_probs * advantages)
        # adv_loss = torch.mean(-log_probs.view(-1, 1) * advantages.view(-1, 1))

        # actor_loss = adv_loss + self.entropy_coeff * entropy
        m1 = Categorical(probs)
        logprobs = m1.log_prob(actions).sum(dim=1).view(-1, 1)
        m2 = Categorical(old_probs)
        old_logprobs = m2.log_prob(actions).sum(dim=1).view(-1, 1)

        ratios = torch.exp(logprobs - old_logprobs.detach())
        surr1 = ratios * advantages.view(-1, 1)
        surr2 = torch.clamp(ratios, 1-self.clip_ratio, 1+self.clip_ratio) * advantages.view(-1, 1)
        # final loss of clipped objective PPO
        actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coeff * m1.entropy().mean()

        self.actor_loss_history.append(actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 10)
        self.actor_optimizer.step()

    def critic_update(self, values, target):

        # Critic update with MSE loss
        # critic_loss = torch.mean(torch.stack([F.mse_loss(value.view(-1), self.td_target_batch[i].view(-1))
        #                                       for i, value in enumerate(self.value_batch)]))
        # critic_loss = torch.mean(torch.stack([torch.mean(adv).pow(2) for adv in self.advantage_batch]))
        # critic_loss = torch.mean(torch.stack([torch.mean(v - self.episode_return_batch[i]).pow(2)
        #                                       for i, v in enumerate(self.value_batch)]))

        # F.mse_loss(self.value_batch, self.td_target_batch)
        diff = values.view(target.shape[0], -1) - target.view(-1, 1)
        critic_loss = (diff * diff).mean()

        self.critic_loss_history.append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 10)
        self.critic_optimizer.step()

    def evaluate(self, path: str, active_branches: list[str] = None, max_actions: int = 5):
        config = {"render_mode": True, "path": path, "max_actions": max_actions, "n_agents": len(active_branches)}
        test_env = RLLibBranchEnv(config)
        state, info = test_env.reset()

        if active_branches is not None:
            test_env.active_branches = active_branches

        state = test_env.get_observation()

        terminated = False
        while not terminated:
            state = self.state_to_tensor(state, test_env)

            distribution, action_dict = self.select_action(state)
            state, reward, terminated, truncated, _ = test_env.step(action_dict)
            print(f"action: {action_dict} \nreward: {reward}\n\n")
            terminated = terminated["__all__"]

    def full_evaluation(self, path: str, n_agents: int = 3, max_actions: int = 5, limit: int = None,
                        plot: bool = False, edges_per_agent: int = 3):

        trial_rewards = []
        test_env = EdgeAgentBranchEnv(path=path, max_actions=max_actions, n_agents=n_agents, agents=["1"])

        counter = 0

        n_buses = len(list(test_env.network_manager.network["bus"].keys()))

        adj_dict = {n: get_adjacent_branches(test_env.network_manager.network, bus_ids=[str(n + 1)])[0]
                    for n in range(n_buses // 2)}

        nodes_with_enough_edges = [b for b in adj_dict.keys() if len(adj_dict[b]) >= edges_per_agent]

        for active_node_list in itertools.permutations(nodes_with_enough_edges, 1):

            active_branches = np.random.choice(adj_dict[active_node_list[0]], size=edges_per_agent, replace=False)

            if limit is not None:
                counter += 1
                if counter >= limit:
                    break

            test_env.set_active_agents(list(active_branches))
            state, info = test_env.reset()

            # state = test_env.get_observation()
            terminated = False

            trial_rewards.append(0)

            while not terminated:
                state = self.state_to_tensor(state, test_env)

                dist, action_dict = self.select_action(state)
                state, reward, terminated, truncated, _ = test_env.step(action_dict)
                print(f"action: {action_dict} \nreward: {reward}\n\n")
                trial_rewards[-1] += np.mean(list(reward.values()))

        if plot:
            plt.figure()
            plt.hist(trial_rewards)
            plt.figure()
            plt.plot(trial_rewards, np.zeros(len(trial_rewards)), 'x')
            print(
                f"Mean trial reward: {np.mean(trial_rewards)}\nMax trial reward: {max(trial_rewards)}\nMin trial reward: {min(trial_rewards)}")
            plt.show()

        return trial_rewards

    def agent_count_evaluation_sweep(self, test_case: str, min_agents: int = 1, max_agents: int = 5,
                                     max_actions: int = 5, limit:int = 100):

        plt_save_info = self.get_plot_string_info()

        n_plots = max_agents - min_agents + 1
        fig_bar, axs_bar = plt.subplots(n_plots, 1, sharex=True, squeeze=False)
        fig_scatter, axs_scatter = plt.subplots(n_plots, 1, sharex=True, squeeze=False)

        fig_bar.supxlabel("Test Case Rewards")
        fig_scatter.supxlabel("Test Case Rewards")

        max_n_agent = max_agents

        for i, agent_count in enumerate(list(range(min_agents, max_n_agent + 1))):
            r = self.full_evaluation(os.path.abspath(test_case), n_agents=agent_count, plot=False,
                                     max_actions=max_actions, limit=limit)
            mean_val = np.mean(r)

            # Histogram plot.
            axs_bar[i, 0].hist(r, density=False)
            axs_bar[i, 0].scatter(mean_val, 0, marker='x', color='r')

            # Error bar plot.
            min_gap = abs(mean_val - min(r))
            max_gap = abs(mean_val - max(r))
            axs_scatter[i, 0].errorbar(mean_val, 0, xerr=[[min_gap], [max_gap]], fmt="o")
            axs_scatter[i, 0].get_yaxis().set_visible(False)

        fig_bar.suptitle(f"Reward histograms: n_agents = {min_agents}-{max_agents} | {test_case}")

        fig_bar.savefig(f"{self.save_directory}a2c_bar_plot_{plt_save_info}.png")
        fig_scatter.savefig(f"{self.save_directory}a2c_eval_return_scatter_{plt_save_info}.png")

    def plot_history(self, additional_text: str = ""):

        plt_file_info = self.get_plot_string_info()

        fig, axs = plt.subplots(3, 1, sharex=True, squeeze=False)
        axs[0, 0].plot(self.reward_history)
        axs[1, 0].plot(np.convolve(self.reward_history, np.ones(100) / 100, mode='same'))
        axs[2, 0].plot(self.episode_length_history)
        fig.suptitle("Episode reward and length history")
        fig.supxlabel("Episode #")
        axs[0, 0].set_ylabel("Episode Reward")
        axs[1, 0].set_ylabel("Averaged Episode Reward")
        axs[2, 0].set_ylabel("Episode Duration")

        file_name = "a2c_reward_duration_train_plot_" + plt_file_info + ".png"
        fig.savefig(f"{self.save_directory}{file_name}")

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.actor_loss_history)
        axs[1].plot(self.critic_loss_history)
        fig.suptitle("Actor critic loss history" + f" {additional_text}")
        fig.supxlabel("# Actor/Critic Updates")
        axs[0].set_ylabel("Actor Loss")
        axs[1].set_ylabel("Critic Loss")

        file_name = "a2c_actor_critic_loss_plot_" + plt_file_info + ".png"
        fig.savefig(f"{self.save_directory} {file_name}")
        # plt.show()

    def save_model(self):

        if not os.path.exists(self.save_directory):
            os.makedirs(f"{self.save_directory}")
        torch.save(self.actor, f"{self.save_directory}a2c_actor.pth")
        torch.save(self.critic, f"{self.save_directory}a2c_critic.pth")

        save_string = ""
        for attribute in [a for a in dir(self) if not a.startswith('__')]:
            save_string += f"{attribute}: {getattr(self, attribute)}\n\n"

        with open(f"{self.save_directory}params.txt", "w") as text_file:
            text_file.write(save_string)

        # Picking - lazy but possibly effective.
        with open(f"{self.save_directory}trainer.pkl", "wb") as f:
            pickle.dump(self, f)

    def load_latest_model(self):

        path = f"./pytorch_models/{self.actor.__class__.__name__}/"

        files = os.listdir(path)
        actor_paths = [os.path.join(path, basename) for basename in files if "actor" in basename]
        critic_paths = [os.path.join(path, basename) for basename in files if "critic" in basename]
        latest_actor_model = max(actor_paths, key=os.path.getctime)
        latest_critic_model = max(critic_paths, key=os.path.getctime)

        self.actor.load_state_dict(torch.load(latest_actor_model, weights_only=True))
        self.critic.load_state_dict(torch.load(latest_critic_model, weights_only=True))

    def state_to_tensor(self, state: dict[str, Union[dict, np.ndarray]], env):

        if isinstance(env, SampledNodeEnv):
            state = {agent: torch.tensor(list(state[agent]), dtype=torch.float32, device=self.device)
                     for agent in env.agents}
        else:
            state = {agent: torch.tensor([state[agent]], dtype=torch.float32, device=self.device)
                     for agent in env.agents}

        return state

    def get_plot_string_info(self):
        plt_file_info = (str(self.n_agents) + "_agents_"
                         + str(self.actor.__class__.__name__) + "_"
                         + str(self.entropy_coeff) + "entropy_"
                         + str(self.batch_size) + "_batch_size_"
                         + time.strftime("%Y%m%d%H%M%S"))

        return plt_file_info


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
                       # "weights": [3, 3, 3, 3]
                       "weights": [1, 1, 0, 0]
                       }
    use_lstm = False
    # load_from_memory = True
    load_from_memory = False
    max_actions = 5

    print(f"LSTM = {use_lstm}")
    trainer = A2CBranchTrainer(actor_type=BusAttentionActor,
                               critic_type=BusAttentionCritic,
                               env_sampling_config=sampler_options,
                               model_linear_dim=128,
                               model_attn_dim=512,
                               max_actions=max_actions,
                               n_agents=5,
                               device="cuda:0",
                               batch_size=64,
                               entropy_coeff=0.01)

    if load_from_memory:
        trainer.load_latest_model()
        results = None
    else:
        results = trainer.train(10000)

    test_case = "ieee_data/WB5.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=5, max_actions=max_actions)

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
