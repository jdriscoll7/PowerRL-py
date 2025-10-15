import itertools
import os
import time
from itertools import count
from typing import Union, Dict

import numpy as np
import torch
from matplotlib import pyplot as plt
from torch import nn, optim, Tensor
import torch.nn.functional as F
from torch.distributions import Categorical

from rl_power.envs.old.rllib_multi_branch_env import RLLibBranchEnv
from rl_power.envs.old.time_varying_branch_env import TVBranchEnv
from rl_power.modules.bus_attention_model import BusAttentionActor, BusAttentionCritic


class A2CTVBranchTrainer:
    def __init__(self, actor_type, critic_type, model_linear_dim: int = 256,
                 model_attn_dim: int = 16,
                 max_actions: Union[int, list] = 10, n_agents: int = 2, device: str = "cpu", batch_size: int = 256,
                 entropy_coeff: float = 0.05, n_heads: int = 4, lr: float = 1e-3, tv_environment: TVBranchEnv = None):

        self.max_actions = max_actions
        self.n_agents = n_agents
        self.discount_rate = 0.99
        self.lr = lr
        self.steps_done = 0
        self.device = device
        self.n_actions = 5
        self.batch_size = batch_size
        self.entropy_coeff = entropy_coeff

        self.state_batch = []
        self.value_batch = []
        self.next_value_batch = []
        self.probs_batch = []
        self.log_probs_batch = []
        self.td_target_batch = []
        self.advantage_batch = []

        self.criterion = nn.SmoothL1Loss()
        self.env = tv_environment

        # Get the number of state observations
        state, info = self.env.reset()
        first_agent = list(state.keys())[0]
        first_branch = list(state.keys())[0]
        self.n_observations = len(state[first_branch])

        self.actor = actor_type(state_length=self.n_observations, n_actions=self.n_actions,
                                attention_embed_dim=model_attn_dim,
                                linear_dim=model_linear_dim, device=self.device, n_heads=n_heads)
        self.critic = critic_type(state_length=self.n_observations, linear_dim=model_linear_dim,
                                  attention_embed_dim=model_attn_dim,
                                  device=self.device, n_heads=n_heads)
        self.actor.to(self.device)
        self.critic.to(self.device)

        self.actor_optimizer = optim.AdamW(self.actor.parameters(), lr=self.lr, amsgrad=True)
        self.critic_optimizer = optim.AdamW(self.critic.parameters(), lr=self.lr, amsgrad=True)

        self.reward_history = []
        self.episode_length_history = []
        self.critic_loss_history = []
        self.actor_loss_history = []

    def select_action(self, state: Dict[str, Tensor]):
        self.steps_done += 1

        action_probs = {agent: torch.zeros(size=(len(branch_data), 4)) for agent, branch_data in state.items()}
        action_dict = {agent: torch.zeros(size=(len(state[agent]),)) for agent, branch_data in state.items()}
        dist = {agent: None for agent, branch_data in state.items()}
        for agent, branch_states in state.items():
            action_probs[agent] = self.actor(branch_states)
            dist[agent] = Categorical(action_probs[agent])
            action_dict[agent] = dist[agent].sample()

        # action_vector = {b: int(policy_output[i].item()) for i, b in enumerate(branches)}

        return dist, action_dict

    def train(self, n_iterations: int = 100):

        # start = time.time()

        for i_episode in range(n_iterations):
            # Initialize the environment and get its state
            state, info = self.env.reset()
            state = {agent: torch.tensor(obs, dtype=torch.float32, device=self.device)
                     for agent, obs in state.items()}

            if i_episode % 100 == 0:
                print(f"Average reward as of episode {i_episode} (last 100 episodes):" + str(np.mean(self.reward_history[-100:])))
                # end = time.time()
                # print(f"Elapsed time: {end-start}")
                # start = end

            self.episode_length_history.append(0)
            self.reward_history.append(0)

            for t in count():

                distribution, action_dict = self.select_action(state)

                next_state, reward, terminated, truncated, _ = self.env.step(action_dict)

                # Convert next state from ndarray dict to tensor dict.
                next_state = {agent: torch.tensor(obs, dtype=torch.float32, device=self.device)
                              for agent, obs in next_state.items()}

                truncated = terminated
                state_tensor = torch.stack(list(state.values()), dim=0)
                next_state_tensor = torch.stack(list(state.values()), dim=0)

                reward_list = [torch.tensor(r, dtype=torch.float32, device=self.device).unsqueeze(0)
                               for branch, r in reward.items()]
                reward_tensor = torch.tensor(reward_list).mean()

                value = self.critic(state_tensor)
                next_value = self.critic(next_state_tensor)
                td_target = reward_tensor + self.discount_rate * next_value.view(-1, 1) * (1 - terminated)
                advantage = td_target - value.view(-1, 1)

                log_prob_list = []
                prob_list = []
                for i, key in enumerate(list(distribution.keys())):
                    log_prob_list.append(distribution[key].log_prob(action_dict[key]))
                    prob_list.append(distribution[key].probs)
                log_probs = torch.stack(log_prob_list, dim=0)
                probs = torch.stack(prob_list, dim=0)

                batch_full = self.update_batch(state_tensor, value, next_value, log_probs, probs, td_target, advantage)

                if batch_full:
                    self.actor_update()
                    self.critic_update()

                self.episode_length_history[i_episode] += 1
                self.reward_history[i_episode] += np.mean(list(reward.values()))

                done = terminated or truncated
                state = next_state
                if done:
                    # episode_durations.append(t + 1)
                    # plot_durations()
                    break

        time_string = time.strftime("%Y%m%d%H%M%S")
        save_directory = f"./pytorch_models/{self.actor.__class__.__name__}/"
        if not os.path.exists(save_directory):
            os.makedirs(f"{save_directory}")
        torch.save(self.actor.state_dict(), f"{save_directory}a2c_actor_weights_{time_string}.pth")
        torch.save(self.critic.state_dict(), f"{save_directory}a2c_critic_weights_{time_string}.pth")

    def update_batch(self, state, value, next_value, log_probs, probs, td_target, advantage) -> bool:

        if len(self.state_batch) == self.batch_size:
            self.state_batch = []
            self.value_batch = []
            self.next_value_batch = []
            self.log_probs_batch = []
            self.probs_batch = []
            self.td_target_batch = []
            self.advantage_batch = []

        self.state_batch.append(state)
        self.value_batch.append(value)
        self.next_value_batch.append(next_value)
        self.log_probs_batch.append(log_probs)
        self.probs_batch.append(probs)
        self.td_target_batch.append(td_target)
        self.advantage_batch.append(advantage)

        return len(self.state_batch) == self.batch_size

    def actor_update(self):

        stacked_probs = torch.cat(self.probs_batch, dim=0)
        entropy = torch.sum(stacked_probs * stacked_probs.log(), dim=-1).mean()

        adv_loss = torch.mean(torch.stack([torch.mean(-log_probs.view(-1, 1) * self.advantage_batch[i].detach())
                                           for i, log_probs in enumerate(self.log_probs_batch)]))

        actor_loss = adv_loss + self.entropy_coeff * entropy
        self.actor_loss_history.append(actor_loss.item())
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 1)
        self.actor_optimizer.step()

    def critic_update(self):

        # Critic update with MSE loss
        critic_loss = torch.mean(torch.stack([F.mse_loss(value.flatten(), self.td_target_batch[i].detach().flatten())
                                              for i, value in enumerate(self.value_batch)]))

        self.critic_loss_history.append(critic_loss.item())
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), 1)
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
            state = {branch: torch.tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
                     for branch, obs in state.items()}

            distribution, action_dict = self.select_action(state)
            state, reward, terminated, truncated, _ = test_env.step(action_dict)
            print(f"action: {action_dict} \nreward: {reward}\n\n")
            terminated = terminated["__all__"]

    def full_evaluation(self, path: str, n_agents: int = 3, max_actions: int = 5, limit: int = None,
                        plot: bool = False):

        trial_rewards = []
        test_env = TVBranchEnv(path=path, max_actions=max_actions, n_agents=n_agents, network_controller=self.env.network_controller)

        counter = 0

        n_buses = len(list(test_env.network_manager.network["bus"].keys()))
        for active_node_list in itertools.combinations([str(b + 1) for b in range(n_buses // 2)], n_agents):

            if limit is not None:
                counter += 1
                if counter >= limit:
                    break

            test_env.agents = active_node_list
            state, info = test_env.reset()

            # state = test_env.get_observation()
            terminated = False

            trial_rewards.append(0)

            while not terminated:
                state = {agent: torch.tensor(obs, dtype=torch.float32, device=self.device)
                         for agent, obs in state.items()}

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
                                     max_actions: int = 5):

        plt_save_info = self.get_plot_string_info()

        n_plots = max_agents - min_agents + 1
        fig_bar, axs_bar = plt.subplots(n_plots, 1, sharex=True)
        fig_scatter, axs_scatter = plt.subplots(n_plots, 1, sharex=True)

        fig_bar.supxlabel("Test Case Rewards")
        fig_scatter.supxlabel("Test Case Rewards")

        max_n_agent = max_agents

        for i, agent_count in enumerate(list(range(min_agents, max_n_agent + 1))):
            r = self.full_evaluation(os.path.abspath(test_case), n_agents=agent_count, plot=False,
                                     max_actions=max_actions)
            mean_val = np.mean(r)

            # Histogram plot.
            axs_bar[i].hist(r, density=False)
            axs_bar[i].scatter(mean_val, 0, marker='x', color='r')

            # Error bar plot.
            min_gap = abs(mean_val - min(r))
            max_gap = abs(mean_val - max(r))
            axs_scatter[i].errorbar(mean_val, 0, xerr=[[min_gap], [max_gap]], fmt="o")
            axs_scatter[i].get_yaxis().set_visible(False)

        fig_bar.suptitle(f"Reward histograms: n_agents = {min_agents}-{max_agents} | {test_case}")

        fig_bar.savefig(f"./figures/a2c_bar_plot_{plt_save_info}.png")
        fig_scatter.savefig(f"./figures/a2c_eval_return_scatter_{plt_save_info}.png")

    def plot_history(self, additional_text: str = ""):

        plt_file_info = self.get_plot_string_info()

        fig, axs = plt.subplots(3, 1, sharex=True)
        axs[0].plot(self.reward_history)
        axs[1].plot(np.convolve(self.reward_history, np.ones(100)/100, mode='same'))
        axs[2].plot(self.episode_length_history)
        fig.suptitle("Episode reward and length history")
        fig.supxlabel("Episode #")
        axs[0].set_ylabel("Episode Reward")
        axs[1].set_ylabel("Averaged Episode Reward")
        axs[2].set_ylabel("Episode Duration")

        file_name = "a2c_reward_duration_train_plot_" + plt_file_info + ".png"
        fig.savefig("./figures/" + file_name)

        fig, axs = plt.subplots(2, 1)
        axs[0].plot(self.actor_loss_history)
        axs[1].plot(self.critic_loss_history)
        fig.suptitle("Actor critic loss history" + f" {additional_text}")
        fig.supxlabel("# Actor/Critic Updates")
        axs[0].set_ylabel("Actor Loss")
        axs[1].set_ylabel("Critic Loss")

        file_name = "a2c_actor_critic_loss_plot_" + plt_file_info + ".png"
        fig.savefig("./figures/" + file_name)
        # plt.show()

    def load_latest_model(self):

        path = f"./pytorch_models/{self.critic.__class__.__name__}/"

        files = os.listdir(path)
        actor_paths = [os.path.join(path, basename) for basename in files if "actor" in basename]
        critic_paths = [os.path.join(path, basename) for basename in files if "critic" in basename]
        latest_actor_model = max(actor_paths, key=os.path.getctime)
        latest_critic_model = max(critic_paths, key=os.path.getctime)

        self.actor.load_state_dict(torch.load(latest_actor_model, weights_only=True))
        self.critic.load_state_dict(torch.load(latest_critic_model, weights_only=True))

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
                               device="cpu",
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
