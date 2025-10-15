import copy
import os
from collections import defaultdict
from typing import Dict, Any, List, Union

import numpy as np
from gymnasium import spaces, Env
import networkx as nx
import random

from rl_power.power.drawing import PMSolutionRenderer
from rl_power.power.graph_utils import get_adjacent_branches, powermodel_dict_to_graph
from rl_power.power.powermodels_interface import Configuration, load_test_case, ConfigurationManager


class NetworkValueController:
    def __init__(self, gen_cost_mean_var: list[float, callable], load_mean_var: list[float, callable]):
        self.gen_cost_mean_var = gen_cost_mean_var
        self.load_mean_var = load_mean_var

    def gen_update(self, generator_data: dict, time: int):

        if len(generator_data["cost"]) == 0:
            return generator_data

        if callable(self.gen_cost_mean_var[0]):
            _mean = self.gen_cost_mean_var[0](time)
        else:
            _mean = self.gen_cost_mean_var[0]

        if callable(self.gen_cost_mean_var[1]):
            _std = self.gen_cost_mean_var[1](time)
        else:
            _std = self.gen_cost_mean_var[1]

        generator_data["cost"][0] += np.random.normal(loc=_mean, scale=_std)
        generator_data["cost"][0] = max(0, generator_data["cost"][0])

        return generator_data

    def load_update(self, load_data: dict, time: int):

        _mean = self.load_mean_var[0]
        _std = self.load_mean_var[1]

        if callable(self.load_mean_var[0]):
            _mean = self.load_mean_var[0](time)
        else:
            _mean = self.load_mean_var[0]

        if callable(self.load_mean_var[1]):
            _std = self.load_mean_var[1](time)
        else:
            _std = self.load_mean_var[1]

        load_data["qd"] += np.random.normal(loc=_mean, scale=_std)
        load_data["pd"] += np.random.normal(loc=_mean, scale=_std)
        load_data["pd"] = max(0, load_data["pd"])
        load_data["qd"] = max(0, load_data["qd"])

        return load_data

    def step(self, network_data: dict, time: int):

        # Update generator data one-by-one.
        for gen_id, gen_data in network_data["gen"].items():
            network_data["gen"][gen_id] = self.gen_update(gen_data, time)

        # Update load data one-by-one.
        for load_id, load_data in network_data["load"].items():
            network_data["load"][load_id] = self.load_update(load_data, time)

        return network_data


class TVBranchEnv(Env):

    def __init__(self, render_mode: str = None, path: str = None, network: dict = None, agents: list[int] = None,
                 max_actions: int = 10, n_agents: int = 3, network_controller: NetworkValueController = None,):

        super().__init__()

        self.network_controller = network_controller
        self.observation_size = 100
        self.last_feasible = True
        self.last_action = None
        self.n_agents = n_agents

        assert path is not None or network is not None
        if path is not None:
            self.network_manager = ConfigurationManager(load_test_case(path))
        else:
            self.network_manager = ConfigurationManager(network)

        n_buses = len(list(self.network_manager.network["bus"].keys())) // 2

        self.n_iterations = 0
        self.last_reward = 0
        self.max_actions = max_actions

        if agents is None:
            agents = np.random.choice([str(i) for i in range(1, n_buses + 1)], size=self.n_agents, replace=False)

        self.agents = agents
        # self.observation_size = observation_size
        self.last_cost = self.network_manager.solution["objective"]

        k = random.randint(1, n_agents)
        # k = n_agents
        self.branches = list(self.network_manager.network["branch"].keys())

        # Four types of configurations for bus-oriented branch configuration.
        # self.action_space = spaces.Dict({x: spaces.Discrete(8) for x in self.branches})

        self.action_space = spaces.Dict({x: spaces.Discrete(4) for x in self.agents})
        self.base_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=float)
        self.observation_space = spaces.Dict({b: self.base_obs_space for b in self.agents})
        self.previous_observation = None

        self.render_mode = render_mode
        if render_mode:
            self.renderer = PMSolutionRenderer()

    def get_observation(self) -> Dict[Any, Any]:

        observations = self.observation_space.sample()

        # Basic graph features for testing.
        current_graph = powermodel_dict_to_graph(self.network_manager.configured_network)
        node_bet = nx.betweenness_centrality(current_graph)
        clust = nx.clustering(current_graph)
        page = nx.pagerank(current_graph)
        jaccard = nx.jaccard_coefficient(current_graph, current_graph.edges)
        jaccard_dict = {(u, v): p for (u, v, p) in jaccard}
        edge_bet = nx.edge_betweenness_centrality(current_graph)

        for b in self.agents:
            one_hot_last_action = np.array([1 if self.last_action[b] == i else 0 for i in
                                            range(5)]) if self.last_action is not None else np.zeros(5)
            running_data = np.array([(self.max_actions - self.n_iterations) / self.max_actions,
                                     int(self.last_feasible)])
            in_bus = self.network_manager.configured_network["branch"][str(b)]["t_bus"]
            out_bus = self.network_manager.configured_network["branch"][str(b)]["f_bus"]

            edge_tuple = (out_bus, in_bus) if (out_bus, in_bus) in jaccard_dict.keys() else (in_bus, out_bus)

            g_feature_vector = np.array([node_bet[in_bus],
                                         node_bet[out_bus],
                                         clust[in_bus],
                                         clust[out_bus],
                                         page[in_bus],
                                         page[out_bus],
                                         jaccard_dict[edge_tuple],
                                         edge_bet[edge_tuple]])

            observations[b] = np.concatenate([self.network_manager.get_branch_state(str(b)).values,
                                                 g_feature_vector,
                                                 running_data,
                                                 one_hot_last_action])

        return observations

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.observation_space = spaces.Dict({b: self.base_obs_space for b in self.agents})
        self.network_manager.reset_configuration()
        self.last_cost = self.network_manager.solution["objective"]
        self.previous_observation = None
        self.last_action = None

        observation = self.get_observation()
        info = self.get_info()

        self.n_iterations = 0
        self.last_reward = 0
        self.last_feasible = True

        return observation, info

    def step(self, action):

        original_cost = self.network_manager.solution["objective"]
        saved_nm = copy.deepcopy(self.network_manager)

        # Compute immediate cost from action and cost from action once network changes.
        left_cost = self.network_manager.solve_branch_configurations(action)
        termination_status_left = str(self.network_manager.config_solution['termination_status']).lower()

        self.update_network_state(self.network_manager.configured_network)

        # Produce right costs to compare new configuration to previous one (cost of each).
        if self.last_action is None:
            last_action = {k: 0 for k, v in action.items()}
            right_pre_cost = self.network_manager.solve_branch_configurations(last_action)
        else:
            right_pre_cost = self.network_manager.solve_branch_configurations(self.last_action)

        right_cost = self.network_manager.solve_branch_configurations(action)

        termination_status_right = str(self.network_manager.config_solution['termination_status']).lower()

        feasible_left = ("infeasible" not in termination_status_left
                         and "iteration_limit" not in termination_status_left
                         and 'numerical' not in termination_status_left)

        feasible_right = ("infeasible" not in termination_status_right
                          and "iteration_limit" not in termination_status_right
                          and 'numerical' not in termination_status_right)

        feasible = feasible_left and feasible_right

        self.last_feasible = feasible

        if not feasible:
            self.network_manager = saved_nm

        # terminated = (self.n_iterations == self.max_actions - 1) or (self.last_action == action) or (not feasible)
        terminated = (self.n_iterations == self.max_actions - 1)
        # terminated = (self.n_iterations == self.max_actions - 1)

        # Dense reward - combine left and right cost (right cost used for last cost).
        if feasible:
            # reward = (self.last_cost - left_cost) / original_cost
            reward = (0.5*(self.last_cost - left_cost) / original_cost +
                      0.5*(right_pre_cost - right_cost) / original_cost)
        else:
            if not self.last_feasible:
                reward = 0
            else:
                reward = -self.last_reward / 100

        reward *= 100

        self.last_action = action
        observation, info = self.get_observation(), self.get_info()

        rewards = {x: reward for x in self.agents}
        self.last_reward = reward

        self.last_cost = right_cost if feasible else self.last_cost

        if self.render_mode is True:
            self.renderer.update_frame(self.network_manager)

        self.n_iterations += 1
        truncated = terminated

        return observation, rewards, terminated, truncated, info

    def update_network_state(self, network_data: dict):
        self.network_controller.step(network_data, self.n_iterations)
        # pass

    def get_info(self):
        return {}


class TVBranchEnvSampler:

    def __init__(self, env_sampling_config: dict, max_actions: Union[str, int] = 10, n_agents: int = 2):
        self.env_sampling_config = env_sampling_config
        self.max_actions = max_actions
        self.n_agents = n_agents

    def sample(self):

        # Select base environment.
        path = random.choices(self.env_sampling_config['paths'],
                              weights=self.env_sampling_config["weights"])[0]

        if isinstance(self.max_actions, list):
            max_actions = random.randint(self.max_actions[0], self.max_actions[1])
        else:
            max_actions = self.max_actions

        base_env = TVBranchEnv(path=path,
                               max_actions=max_actions,
                               n_agents=self.n_agents)

        # Make random modifications.
        env = self.perturb_power_network(base_env)

        return env

    def perturb_power_network(self, env):
        # Randomly decrease loads.

        # Randomly change generator costs.

        # For now do nothing...
        return env


class SampledTVNodeEnv(Env):

    def __init__(self, env_sampling_config: dict, max_actions: Union[int, list] = 10, n_agents: int = 2):
        super().__init__()
        self.sampler = TVBranchEnvSampler(env_sampling_config, max_actions, n_agents)
        self.env = self.sampler.sample()
        self.agents = self.env.agents
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        self.adjacency_list = self.env.adjacency_list

    def get_observation(self) -> Dict[Any, Any]:
        return self.env.get_observation()

    def reset(self, seed=None, options=None):
        self.env = self.sampler.sample()
        self.adjacency_list = self.env.adjacency_list
        self.agents = self.env.agents
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def get_info(self):
        return self.env.get_info()


if __name__ == "__main__":
    path = os.path.abspath("ieee_data/pglib_opf_case30_ieee.m")
    agents = [1]
    max_actions = 10
    n_agents = 1

    test_env = TVBranchEnv(path=path, agents=agents, max_actions=max_actions, n_agents=n_agents)
    a_space = test_env.action_space
    a = a_space.sample()
    o, r, t, _, info = test_env.step(a)
    print(r)
