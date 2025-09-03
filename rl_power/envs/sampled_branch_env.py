import random
from typing import Any, Dict, Union

from rl_power.envs.rllib_multi_branch_env import RLLibBranchEnv


class EnvSampler:

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

        base_env = RLLibBranchEnv({"path": path,
                                   "max_actions": max_actions,
                                   "n_agents": self.n_agents})

        # Make random modifications.
        env = self.perturb_power_network(base_env)

        return env

    def perturb_power_network(self, env):
        # Randomly decrease loads.

        # Randomly change generator costs.

        # For now do nothing...
        return env


class SampledBranchEnv():

    def __init__(self, env_sampling_config: dict, max_actions: Union[int, list] = 10, n_agents: int = 2):
        super().__init__()
        self.sampler = EnvSampler(env_sampling_config, max_actions, n_agents)
        self.env = self.sampler.sample()
        self.active_branches = self.env.active_branches
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space

    def get_observation(self) -> Dict[Any, Any]:
        return self.env.get_observation()

    def reset(self, seed=None, options=None):
        self.env = self.sampler.sample()
        self.active_branches = self.env.active_branches
        self.action_space = self.env.action_space
        self.observation_space = self.env.observation_space
        return self.env.reset(seed=seed, options=options)

    def step(self, action):
        return self.env.step(action)

    def get_info(self):
        return self.env.get_info()
