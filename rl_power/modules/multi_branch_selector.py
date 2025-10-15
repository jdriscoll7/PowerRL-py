import os
from typing import Tuple, Callable

import torch
from gymnasium import spaces
from stable_baselines3 import A2C
from stable_baselines3.common.policies import ActorCriticPolicy
from torch import nn

from rl_power.envs.old.grouped_branch_env import GroupedBranchEnv


class Transpose(nn.Module):
    def forward(self, x):
        return torch.transpose(x, -1, -2)


class MultiBranchSelector(nn.Module):
    def __init__(
            self,
            feature_dim: int,
            n_branches: int,
            groups: list[list[str]],
            last_layer_dim_pi: int = 64,
            last_layer_dim_vf: int = 64,
    ):
        super().__init__()

        self.groups = groups
        self.n_branches = n_branches
        feature_dim = feature_dim // n_branches

        # IMPORTANT:
        # Save output dimensions, used to create the distributions
        self.latent_dim_pi = last_layer_dim_pi
        self.latent_dim_vf = last_layer_dim_vf

        # Policy network
        self.policy_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_pi), nn.ReLU(), Transpose(), nn.Linear(self.n_branches, 1),  nn.ReLU(), nn.Flatten(-2, -1)
        )
        # Value network
        self.value_net = nn.Sequential(
            nn.Linear(feature_dim, last_layer_dim_vf), nn.ReLU(), Transpose(), nn.Linear(self.n_branches, 1),  nn.ReLU(), nn.Flatten(-2, -1)
        )

    def forward(self, features: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.forward_actor(features), self.forward_critic(features)

    def forward_actor(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        return self.policy_net(features.reshape(shape=(batch_size, self.n_branches, -1)))

    def forward_critic(self, features: torch.Tensor) -> torch.Tensor:
        batch_size = features.shape[0]
        return self.value_net(features.reshape(shape=(batch_size, self.n_branches, -1)))


class MultiBranchSelectorPolicy(ActorCriticPolicy):
    def __init__(
            self,
            observation_space: spaces.Space,
            action_space: spaces.Space,
            lr_schedule: Callable[[float], float],
            *args,
            **kwargs,
    ):
        # Disable orthogonal initialization
        kwargs["ortho_init"] = False
        self.groups = kwargs.get("groups")
        self.n_branches = kwargs.get("n_branches")

        super().__init__(
            observation_space,
            action_space,
            lr_schedule,
            # Pass remaining arguments to base class
        )

        # self.action_net =

    def _build_mlp_extractor(self) -> None:
        self.mlp_extractor = MultiBranchSelector(feature_dim=self.features_dim,
                                                 groups=self.groups,
                                                 n_branches=self.n_branches)


if __name__ == "__main__":
    nv_options = {"path": os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
                  "groups": None}

    test_env = GroupedBranchEnv(nv_options, )

    policy_args = {"n_branches": len(test_env.branches),
                   "groups": test_env.groups}

    n_iterations = 5000
    log_interval = 1

    # model = PPO(MultiBranchSelectorPolicy, test_env, verbose=1, policy_kwargs=policy_args, stats_window_size=log_interval)
    model = A2C(MultiBranchSelectorPolicy, test_env, verbose=1, policy_kwargs=policy_args, stats_window_size=log_interval)
    model.learn(5000, log_interval=log_interval)
    # model.learn(5000, )
    model.save(f"trained_models/branch_env_model_{n_iterations}")
