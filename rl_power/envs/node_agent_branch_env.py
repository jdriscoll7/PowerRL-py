import copy
import os
from collections import defaultdict
from typing import Dict, Any, List, Union

import numpy as np
from gymnasium import spaces, Env
import networkx as nx
import random

from rl_power.envs.time_varying_branch_env import NetworkValueController
from rl_power.power.drawing import PMSolutionRenderer
from rl_power.power.graph_utils import get_adjacent_branches, powermodel_dict_to_graph
from rl_power.power.powermodels_interface import Configuration, load_test_case, ConfigurationManager, solve_opf


class NodeAgentBranchEnv(Env):

    def __init__(self, render_mode: str = None, path: str = None, network: dict = None, agents: list[int] = None,
                 max_actions: int = 10, n_agents: int = 3):

        super().__init__()

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
        self.adjacency_list = {x: get_adjacent_branches(self.network_manager.network,
                                                        [str(x), str(self.network_manager.get_busbar_id(x))])[0] for x
                               in self.agents}
        self.action_space = spaces.Dict({x: spaces.Discrete(4) for x in self.agents})
        self.base_obs_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.observation_size,), dtype=float)
        self.observation_space = spaces.Dict(
            {x: spaces.Dict({b: self.base_obs_space for b in self.adjacency_list[x]}) for x in self.agents})
        self.previous_observation = None

        self.render_mode = render_mode
        if render_mode:
            self.renderer = PMSolutionRenderer()

    def get_observation(self) -> Dict[Any, Any]:

        # observations = {b: np.zeros(self.observation_size * 2) for b in self.active_branches}
        self.adjacency_list = {x: get_adjacent_branches(self.network_manager.network,
                                                        [str(x), str(self.network_manager.get_busbar_id(x))])[0] for x
                               in self.agents}
        observations = self.observation_space.sample()

        # Basic graph features for testing.
        current_graph = powermodel_dict_to_graph(self.network_manager.configured_network)
        node_bet = nx.betweenness_centrality(current_graph)
        clust = nx.clustering(current_graph)
        page = nx.pagerank(current_graph)
        jaccard = nx.jaccard_coefficient(current_graph, current_graph.edges)
        jaccard_dict = {(u, v): p for (u, v, p) in jaccard}
        edge_bet = nx.edge_betweenness_centrality(current_graph)

        for n in self.agents:

            connected_branches = self.adjacency_list[n]

            for b in connected_branches:
                one_hot_last_action = np.array([1 if self.last_action[b] == i else 0 for i in
                                                range(5)]) if self.last_action is not None else np.zeros(5)
                running_data = np.array([(self.max_actions - self.n_iterations) / self.max_actions,
                                         int(self.last_feasible)])
                in_bus = self.network_manager.configured_network["branch"][b]["t_bus"]
                out_bus = self.network_manager.configured_network["branch"][b]["f_bus"]

                edge_tuple = (out_bus, in_bus) if (out_bus, in_bus) in jaccard_dict.keys() else (in_bus, out_bus)

                g_feature_vector = np.array([node_bet[in_bus],
                                             node_bet[out_bus],
                                             clust[in_bus],
                                             clust[out_bus],
                                             page[in_bus],
                                             page[out_bus],
                                             jaccard_dict[edge_tuple],
                                             edge_bet[edge_tuple]])

                observations[n][b] = np.concatenate([
                                                     self.network_manager.get_branch_state(str(b)).values,
                                                     g_feature_vector,
                                                     # running_data,
                                                     one_hot_last_action,
                                                     [-1 if b == str(in_bus) else 1]])


        return observations

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.adjacency_list = {x: get_adjacent_branches(self.network_manager.network, [str(x)])[0] for x in self.agents}
        self.observation_space = spaces.Dict(
            {x: spaces.Dict({b: self.base_obs_space for b in self.adjacency_list[x]}) for x in self.agents})
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

        # Map bus-local actions to branch-local actions.
        # branch_bus_map = defaultdict(list)
        # for k, v in self.adjacency_list.items():
        #     for branch in v:
        #         branch_bus_map[branch].append(k)
        # action = {b: local_bus_branch_map(self.network_manager.network, b, random.choice(branch_bus_map[b]), a) for b, a in action.items()}
        action = convert_bus_actions_to_branch(self.network_manager.network, action, self.adjacency_list)

        original_cost = self.network_manager.solution["objective"]
        saved_nm = copy.deepcopy(self.network_manager)
        new_cost = self.network_manager.solve_branch_configurations(action)

        termination_status = str(self.network_manager.config_solution['termination_status']).lower()

        feasible = ("infeasible" not in termination_status
                    and "iteration_limit" not in termination_status
                    and 'numerical' not in termination_status)

        self.last_feasible = feasible

        if not feasible:
            self.network_manager = saved_nm

        # reward = self.last_cost - new_cost if feasible else -self.network_manager.solution["objective"]
        # reward = self.network_manager.solution["objective"]-new_cost if feasible else -self.network_manager.solution["objective"]
        # reward = (self.last_cost - new_cost) / original_cost if feasible else -0.001
        # rewards = {x: reward for x in self.branches}

        # terminated = (self.n_iterations == self.max_actions - 1) or (self.last_action == action) or (not feasible)
        terminated = (self.n_iterations == self.max_actions - 1) or (self.last_action == action)
        # terminated = (self.n_iterations == self.max_actions - 1)

        # Sparse reward.
        # if terminated:
        #     reward = (original_cost - new_cost) / original_cost if feasible else 0
        # else:
        #     reward = 0

        # Dense reward.
        scale_factor = 1
        reward = (self.last_cost - new_cost) / original_cost if feasible else 0
        reward *= scale_factor

        # Binary reward.
        # if self.last_cost > new_cost + 1e-5 and feasible:
        #     reward = 1
        # elif self.last_cost == new_cost:
        #     reward = 0
        # # elif self.last_cost < new_cost or not feasible:
        # #     reward = -1
        # else:
        #     reward = -1
        self.last_improvement = (original_cost - new_cost) / original_cost if feasible else 0
        self.last_improvement *= 100

        self.last_action = action
        observation, info = self.get_observation(), self.get_info()

        rewards = {x: reward for x in self.agents}
        self.last_reward = reward

        self.last_cost = new_cost if feasible else self.last_cost

        if self.render_mode is True:
            self.renderer.update_frame(self.network_manager)

        self.n_iterations += 1
        truncated = terminated

        return observation, rewards, terminated, truncated, info

    def get_info(self):
        return {}


class NodeEnvSampler:

    def __init__(self, env_sampling_config: dict, max_actions: Union[str, int] = 10, n_agents: int = 2):
        self.env_sampling_config = env_sampling_config
        self.max_actions = max_actions
        self.n_agents = n_agents
        self.n_samples = 0

        self.load_mean_var = self.env_sampling_config.get("load_mean_var", [0, 1])
        self.gen_cost_mean_var = self.env_sampling_config.get("gen_cost_mean_var", [0, 1])

        self.net_controller = NetworkValueController(gen_cost_mean_var=self.gen_cost_mean_var, load_mean_var=self.load_mean_var)

    def sample(self):

        # Select base environment.
        path = random.choices(self.env_sampling_config['paths'],
                              weights=self.env_sampling_config["weights"])[0]

        if isinstance(self.max_actions, list):
            max_actions = random.randint(self.max_actions[0], self.max_actions[1])
        else:
            max_actions = self.max_actions

        base_env = NodeAgentBranchEnv(path=path,
                                      max_actions=max_actions,
                                      n_agents=self.n_agents)

        # Make random modifications.
        env = self.perturb_power_network(base_env)

        self.n_samples += 1

        return env

    def perturb_power_network(self, env):

        # Perturb power network problem parameters.
        env.network_manager.network = self.net_controller.step(env.network_manager.network, self.n_samples)
        env.network_manager.solution = solve_opf(env.network_manager.network)
        env.network_manager.reset_configuration()

        return env


class SampledNodeEnv(Env):

    def __init__(self, env_sampling_config: dict, max_actions: Union[int, list] = 10, n_agents: int = 2):
        super().__init__()
        self.sampler = NodeEnvSampler(env_sampling_config, max_actions, n_agents)
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


def local_bus_branch_map(network: dict, bus_id: str, branch_id: str, action: int):
    assert action < 3

    if action == 0:
        action = 0
    elif action == 1:
        is_t_bus = network["branch"][branch_id]["t_bus"] == int(bus_id)
        action = 1 << int(is_t_bus)
    else:
        action = 4

    return action


def node_action_to_branch_action(node_action_dict: dict, adjacency_list: dict):
    branch_action_dict = defaultdict(list, [])
    for agent in list(node_action_dict.keys()):
        for i, branch in enumerate(adjacency_list[agent]):
            branch_action_dict[branch].append(node_action_dict[agent][i].item())

    return branch_action_dict


def convert_bus_actions_to_branch(network: dict, action: dict, adjacency_list: dict):
    # Map bus-local actions to branch-local actions.
    branch_bus_map = defaultdict(list)
    for k, v in adjacency_list.items():
        for branch in v:
            branch_bus_map[branch].append(k)

    # Fill out branch-level actions.
    branch_actions = {b: 0 for b in branch_bus_map.keys()}
    for agent, branch_action_tensor in action.items():
        for i_branch, branch in enumerate(adjacency_list[agent]):
            action_to_add = local_bus_branch_map(network, branch, agent, branch_action_tensor[i_branch])
            branch_actions[branch] |= action_to_add
            if branch_actions[branch] > 4:
                branch_actions[branch] = 4

    return branch_actions


if __name__ == "__main__":
    path = os.path.abspath("ieee_data/pglib_opf_case30_ieee.m")
    agents = [1]
    max_actions = 10
    n_agents = 1

    test_env = NodeAgentBranchEnv(path=path, agents=agents, max_actions=max_actions, n_agents=n_agents)
    a_space = test_env.action_space
    a = a_space.sample()
    o, r, t, _, info = test_env.step(a)
    print(r)
