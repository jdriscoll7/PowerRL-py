import copy
import os
from typing import Union, Dict

import numpy as np
import torch
from torch import Tensor
from torch.distributions import Categorical

from rl_power.envs.edge_agent_branch_env import EdgeAgentBranchEnv
from rl_power.envs.old.node_agent_branch_env import SampledNodeEnv
from rl_power.power.drawing import PMSolutionRenderer
from rl_power.power.readable_actions import action_branch_data_to_readable


class A2CBranchTester:
    def __init__(self, test_env_path: dict, actor_critic_directory: str, n_agents: int = 3, device: str = "cuda:0"):

        self.steps_done = 0
        self.env = EdgeAgentBranchEnv(network=test_env_path,
                                      max_actions=1000,
                                      n_agents=n_agents,
                                      agents=["1"])

        self.initial_cost = self.env.network_manager.solution["objective"]

        self.device = device
        # Get the number of state observations
        state, info = self.env.reset()
        first_agent = list(state.keys())[0]

        if isinstance(self.env, SampledNodeEnv):
            first_branch = list(state[first_agent].keys())[0]
            self.n_observations = len(state[first_agent][first_branch])
        else:
            first_branch = list(state.keys())[0]
            self.n_observations = len(state[first_branch])

        self.actor = None
        self.critic = None
        self.start_renderer = PMSolutionRenderer()
        self.start_renderer.update_frame(self.env.network_manager)
        # plt.show()
        self.start_fig = self.start_renderer.get_fig()

        self.renderer = PMSolutionRenderer(layout=copy.deepcopy(self.start_renderer.layout))
        # self.renderer = PMSolutionRenderer()
        self.renderer.update_frame(self.env.network_manager)
        self.current_fig = self.renderer.get_fig()
        # self.start_fig = self.current_fig
        # plt.show()

        if actor_critic_directory is not None:
            self.load_models(actor_critic_directory)

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

    def test_step(self, agents: list[int] = None):

        self.env.set_active_agents(agents)

        state = self.env.get_observation()
        state = self.state_to_tensor(state, self.env)

        distribution, action = self.select_action(state)

        next_state, reward, terminated, truncated, info = self.env.step(action)

        self.renderer.update_frame(self.env.network_manager, f"| Initial cost: {self.initial_cost:.2f}")
        self.current_fig = self.renderer.get_fig()

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

        n_buses = len(self.env.network_manager.network["bus"].keys())
        action_string = ""
        for i, _branch in enumerate(self.env.agents):
            action_string += action_branch_data_to_readable(action[i],
                                                            self.env.network_manager.configured_network["branch"][_branch],
                                                            n_buses=n_buses//2) + "\n"
        action_string += f"{info['termination_status']}\n"

        return distribution, action_string

    def load_models(self, path):

        files = os.listdir(path)
        actor_paths = [os.path.join(path, basename) for basename in files if "actor" in basename and ".pth" in basename]
        critic_paths = [os.path.join(path, basename) for basename in files if "critic" in basename and ".pth" in basename]
        latest_actor_model = max(actor_paths, key=os.path.getctime)
        latest_critic_model = max(critic_paths, key=os.path.getctime)

        print("actor loading")
        self.actor = torch.load(latest_actor_model, weights_only=False)
        print("actor loaded")
        print("critic loading")
        self.critic = torch.load(latest_critic_model, weights_only=False)
        print("critic loaded")

    def state_to_tensor(self, state: dict[str, Union[dict, np.ndarray]], env):

        if isinstance(env, SampledNodeEnv):
            state = {agent: torch.tensor(list(state[agent]), dtype=torch.float32, device=self.device)
                     for agent in env.agents}
        else:
            state = {agent: torch.tensor([state[agent]], dtype=torch.float32, device=self.device)
                     for agent in env.agents}

        return state


if __name__ == '__main__':
    sampler_options = {"paths": [os.path.abspath("ieee_data/WB5.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case14_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case57_ieee.m")],
                       # "weights": [3, 3, 3, 3]
                       "weights": [1, 1, 0, 0]
                       }
