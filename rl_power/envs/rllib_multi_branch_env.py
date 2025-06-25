import copy
import os
from typing import Dict, Any, List

import numpy as np
import pandas as pd
import pygame
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random
from ray.rllib.env import MultiAgentEnv
from tensorflow.python.autograph.pyct.templates import replace

from rl_power.power.drawing import PMSolutionRenderer
from rl_power.power.powermodels_interface import Configuration, load_test_case, ConfigurationManager


class RLLibBranchEnvBase(MultiAgentEnv):

    def __init__(self, render_mode: str = None, path: str = None, network: dict = None, groups: list[list[int]] = None,
                 max_actions: int = 10, active_branches: list[str] = None, n_agents: int = 3):

        super().__init__()

        self.last_feasible = True
        self.last_improvement = 0
        self.last_action = None
        self.n_agents = n_agents

        assert path is not None or network is not None
        if path is not None:
            self.network_manager = ConfigurationManager(load_test_case(path))
        else:
            self.network_manager = ConfigurationManager(network)

        self.n_iterations = 0
        self.last_reward = 0
        self.max_actions = max_actions

        if groups is None:
            groups = [[b for b in self.network_manager.network["branch"].keys()]]

        self.groups = groups
        observation_size = 20
        self.observation_size = observation_size
        self.last_cost = self.network_manager.solution["objective"]

        k = random.randint(1, n_agents)
        # k = n_agents
        self.branches = list(self.network_manager.network["branch"].keys())

        if active_branches is None:
            self.active_branches = np.random.choice(list(self.network_manager.network["branch"].keys()), k,
                                                    replace=False)
        else:
            self.active_branches = active_branches

        # Four types of configurations for bus-oriented branch configuration.
        # self.action_space = spaces.Dict({x: spaces.Discrete(8) for x in self.branches})
        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=float)
        self.previous_observation = None

        self.render_mode = render_mode
        if render_mode:
            self.renderer = PMSolutionRenderer()

    def get_observation(self) -> Dict[Any, Any]:

        # observations = {b: np.zeros(self.observation_size * 2) for b in self.active_branches}
        observations = {b: np.zeros(self.observation_size) for b in self.active_branches}

        if self.previous_observation is None:
            previous_observation = {b: np.zeros(self.observation_size) for b in self.active_branches}
        else:
            previous_observation = self.previous_observation

        for b in self.active_branches:
            one_hot_last_action = np.array([1 if self.last_action[b] == i else 0 for i in range(5)]) if self.last_action is not None else np.zeros(5)
            running_data = np.array([self.last_improvement,
                                     (self.max_actions - self.n_iterations) / self.max_actions,
                                     int(self.last_feasible)])

            # observations[b] = np.concatenate([self.network_manager.get_branch_state(str(b)).values,
            #                                   running_data,
            #                                   one_hot_last_action])
            observations[b] = np.concatenate([self.network_manager.get_branch_state(str(b)).values[-12:],
                                                 # running_data,
                                                 one_hot_last_action])
            # observations[b][:self.observation_size] = observation_vector
            # observations[b][self.observation_size:] = observation_vector - previous_observation[b]
            # observations[b] = np.concatenate([running_data,
            #                                   one_hot_last_action])

        return observations

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.network_manager.reset_configuration()
        self.last_cost = self.network_manager.solution["objective"]
        self.last_improvement = 0
        self.previous_observation = None

        observation = self.get_observation()
        info = self.get_info()

        self.n_iterations = 0
        self.last_reward = 0
        self.last_action = None
        self.last_feasible = True

        return observation, info

    def step(self, action):

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
        if terminated:
            reward = (original_cost - new_cost) / original_cost if feasible else 0
        else:
            reward = 0

        # Dense reward.
        # reward = (self.last_cost - new_cost) / original_cost if feasible else -0.1
        # reward = (self.last_cost - new_cost) / original_cost if feasible else 0
        # reward = (original_cost - new_cost) / original_cost if feasible else 0

        reward *= 100

        self.last_improvement = (original_cost - new_cost) / original_cost if feasible else 0
        self.last_improvement *= 100

        self.last_action = action
        observation, info = self.get_observation(), self.get_info()
        self.previous_observation = {b: obs[:self.observation_size] for b, obs in observation.items()}

        rewards = {x: reward for x in self.active_branches}
        self.last_reward = reward

        self.last_cost = new_cost if feasible else self.last_cost

        terminateds = {x: terminated for x in self.active_branches}
        terminateds["__all__"] = terminated

        if self.render_mode is True:
            self.renderer.update_frame(self.network_manager)

        self.n_iterations += 1

        truncateds = {x: False for x in self.active_branches}
        truncateds["__all__"] = False

        return observation, rewards, terminateds, truncateds, info

    def get_info(self):
        return {}


class RLLibBranchEnv(RLLibBranchEnvBase):
    def __init__(self, config: dict):
        render_mode = config.get("render_mode")
        path = config.get("path")
        network = config.get("network")
        groups = config.get("groups")
        max_actions = config.get("max_actions")
        n_agents = config.get("n_agents")
        super().__init__(render_mode=render_mode, path=path, network=network, groups=groups, max_actions=max_actions,
                         n_agents=n_agents)

    def draw_state(self):
        pass


if __name__ == "__main__":
    nv_options = {"path": os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
                  "groups": None}

    test_env = RLLibBranchEnv(nv_options, )
    a_space = test_env.action_space
    a = a_space.sample()
    o, r, t, _, info = test_env.step(a)
    print(r)
