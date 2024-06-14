import numpy as np
import pandas as pd
import pygame
import gymnasium as gym
from gymnasium import spaces
import networkx as nx
import random

from rlpower.power.drawing import PMSolutionRenderer
from rlpower.power.powermodels_interface import Configuration, load_test_case, ConfigurationManager


class BranchEnvBase(gym.Env):

    def __init__(self, render_mode: str = None, path: str = None, network: dict = None):
        assert path is not None or network is not None

        if path is not None:
            self.network_manager = ConfigurationManager(load_test_case(path))
        else:
            self.network_manager = ConfigurationManager(network)

        observation_size = self.network_manager.branch_state_length

        self.last_cost = self.network_manager.solution["objective"]

        self.window_size = 512

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(observation_size,), dtype=float)

        # Four types of configurations for bus-oriented branch configuration.
        self.action_space = spaces.Discrete(8)
        self.current_branch = random.choice(list(self.network_manager.network["branch"].keys()))

        self.render_mode = render_mode
        if render_mode:
            self.renderer = PMSolutionRenderer()

    def get_observation(self):
        return self.network_manager.get_branch_state(self.current_branch).values

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        # Choose the agent's location uniformly at random
        self.network_manager.reset_configuration()
        self.last_cost = self.network_manager.solution["objective"]
        self.current_branch = str(random.choice(list(self.network_manager.network["branch"].keys())))

        observation = self.get_observation()
        info = self.get_info()

        return observation, info

    def get_info(self):
        return {}

    def step(self, action):

        new_cost = self.network_manager.solve_branch_configuration(action, str(self.current_branch))
        feasible = "infeasible" not in str(self.network_manager.config_solution['termination_status']).lower()

        terminated = not feasible

        original_cost = self.network_manager.solution["objective"]

        # reward = self.last_cost - new_cost if feasible else -self.network_manager.solution["objective"]
        reward = (self.last_cost - new_cost) / original_cost if feasible else -0.5
        # reward = self.network_manager.solution["objective"]-new_cost if feasible else -self.network_manager.solution["objective"]

        self.last_cost = new_cost

        observation = self.get_observation()
        info = self.get_info()

        self.current_branch = str(random.choice(list(self.network_manager.network["branch"].keys())))

        if self.render_mode is True:
            self.renderer.update_frame(self.network_manager)

        return observation, reward, terminated, False, info


class BranchEnv(BranchEnvBase):
    def __init__(self, config: dict):
        render_mode = config.get("render")
        path = config.get("path")
        network = config.get("network")
        super().__init__(render_mode=render_mode, path=path, network=network)

    def draw_state(self):
        pass
