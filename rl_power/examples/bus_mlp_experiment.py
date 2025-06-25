import os

from rl_power.envs.time_varying_branch_env import NetworkValueController
from rl_power.modules.bus_attention_model import BusAttentionActor, BusAttentionCritic
from rl_power.training.a2c_trainer import A2CBranchTrainer

import torch
from torch import nn
import torch.nn.functional as F


class MLPActor(nn.Module):
    def __init__(self, state_length: int, linear_dim: int = 256, n_actions: int = 8, **kwargs):
        super().__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        self.state_length = state_length
        self.linear_dim = linear_dim

        self.layer_1 = nn.Linear(in_features=state_length, out_features=linear_dim, device=self.device)
        self.layer_2 = nn.Linear(in_features=linear_dim, out_features=linear_dim, device=self.device)
        self.layer_3 = nn.Linear(in_features=linear_dim, out_features=n_actions, device=self.device)

    def forward(self, x):

        x = x.sign() * x.abs().pow(1/7)

        x = self.layer_1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_3(x)

        return F.softmax(x, dim=-1)
        # return x


class MLPCritic(nn.Module):
    def __init__(self, state_length: int, linear_dim: int = 256, n_actions: int = 8, **kwargs):
        super().__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = "cpu"

        self.state_length = state_length
        self.linear_dim = linear_dim

        self.layer_1 = nn.Linear(in_features=state_length, out_features=linear_dim, device=self.device)
        self.layer_2 = nn.Linear(in_features=linear_dim, out_features=linear_dim, device=self.device)
        self.layer_3 = nn.Linear(in_features=linear_dim, out_features=1, device=self.device)

    def forward(self, x):

        x = x.sign() * x.abs().pow(1/7)

        x = self.layer_1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_3(x)

        return x


if __name__ == '__main__':
    variance = lambda t: 1
    sampler_options = {"paths": [os.path.abspath("ieee_data/WB5.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case14_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case57_ieee.m")],
                       # "weights": [3, 1, 0.33, 0.11],
                       # "weights": [1, 1, 1, 0],
                       "weights": [1, 0, 0, 0],
                       "gen_cost_mean_var": [0, variance],
                       "load_mean_var": [0, variance],
                       }

    use_lstm = False
    # load_from_memory = True
    load_from_memory = False
    max_actions = 5
    n_train_agents = 2

    print(f"LSTM = {use_lstm}")
    trainer = A2CBranchTrainer(actor_type=MLPActor,
                               critic_type=MLPCritic,
                               env_sampling_config=sampler_options,
                               model_linear_dim=64,
                               max_actions=max_actions,
                               n_agents=n_train_agents,
                               device="cpu",
                               batch_size=1,
                               entropy_coeff=1e-4,
                               lr=1e-4)

    if load_from_memory:
        trainer.load_latest_model()
        results = None
    else:
        results = trainer.train(50000)

    test_case = "ieee_data/WB5.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=3, max_actions=max_actions)

    trainer.plot_history(f"- Trained on {n_train_agents} Agents")


    # test_case = "ieee_data/pglib_opf_case14_ieee.m"
    # trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=3)
    #
    # test_case = "ieee_data/pglib_opf_case30_ieee.m"
    # trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=3)

    test_case = "ieee_data/pglib_opf_case57_ieee.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=2)

    # trainer.full_evaluation(), n_agents=2)

    # trainer.plot_history()
    # trainer.evaluate(os.path.abspath("ieee_data/pglib_opf_case57_ieee.m"), active_branches=['8'])

    print("")

    # run_policy_on_branch_env(eval_env, algo, 5)