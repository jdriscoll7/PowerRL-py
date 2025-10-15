import os

import numpy as np
import pandas as pd
import pygame
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random

from rl_power.power.drawing import PMSolutionRenderer
from rl_power.power.powermodels_interface import Configuration, load_test_case, ConfigurationManager


class GroupedBranchEnvBase(gym.Env):

    def __init__(self, render_mode: str = None, path: str = None, network: dict = None, groups: list[list[int]] = None,
                 max_actions: int = 10):

        assert path is not None or network is not None
        if path is not None:
            self.network_manager = ConfigurationManager(load_test_case(path))
        else:
            self.network_manager = ConfigurationManager(network)

        self.n_iterations = 0
        self.max_actions = max_actions

        if groups is None:
            groups = [[b] for b in self.network_manager.network["branch"].keys()]
        self.groups = groups

        observation_size = self.network_manager.branch_state_length

        self.last_cost = self.network_manager.solution["objective"]

        self.branches = list(self.network_manager.network["branch"].keys())
        self.branches.sort(key=lambda x: int(x))

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(len(self.branches), observation_size,),
                                            dtype=float)

        # Four types of configurations for bus-oriented branch configuration.
        self.action_space = spaces.MultiDiscrete([8 for x in self.branches])

        self.render_mode = render_mode
        if render_mode:
            self.renderer = PMSolutionRenderer()

    def get_observation(self):

        observation = np.zeros(self.observation_space.shape, dtype=float)

        for b in self.branches:
            observation[int(b) - 1] = self.network_manager.get_branch_state(str(b)).values

        return observation

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.network_manager.reset_configuration()
        self.last_cost = self.network_manager.solution["objective"]

        observation = self.get_observation()
        info = self.get_info()

        self.n_iterations = 0

        return observation, info

    def step(self, action):

        original_cost = self.network_manager.solution["objective"]

        new_cost = self.network_manager.solve_branch_configurations(action, self.branches)

        termination_status = str(self.network_manager.config_solution['termination_status']).lower()

        feasible = ("infeasible" not in termination_status
                    and "iteration_limit" not in termination_status
                    and 'numerical' not in termination_status)


        # reward = self.last_cost - new_cost if feasible else -self.network_manager.solution["objective"]
        # reward = self.network_manager.solution["objective"]-new_cost if feasible else -self.network_manager.solution["objective"]
        reward = (self.last_cost - new_cost) / original_cost if feasible else -0.01

        self.last_cost = new_cost

        observation, info = self.get_observation(), self.get_info()

        terminated = (self.n_iterations == 2)

        if self.render_mode is True:
            self.renderer.update_frame(self.network_manager)

        self.n_iterations += 1

        return observation, reward, terminated, False, info

    def get_info(self):
        return {}


class GroupedBranchEnv(GroupedBranchEnvBase):
    def __init__(self, config: dict):
        render_mode = config.get("render")
        path = config.get("path")
        network = config.get("network")
        groups = config.get("groups")
        super().__init__(render_mode=render_mode, path=path, network=network, groups=groups)

    def draw_state(self):
        pass


if __name__ == "__main__":
    nv_options = {"path": os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
                  "groups": None}

    test_env = GroupedBranchEnv(nv_options, )
    a_space = test_env.action_space
    a = a_space.sample()
    o, r, t, _, info = test_env.step(a)
    print(r)
