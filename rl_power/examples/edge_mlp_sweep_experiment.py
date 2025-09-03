import itertools
import os
from multiprocessing import Pool

from rl_power.envs.edge_agent_branch_env import SampledEdgeEnv
from rl_power.envs.time_varying_branch_env import NetworkValueController
from rl_power.modules.branch_mlp_models import MLPCombinedActor, MLPCombinedCritic, MLPActor, MLPCritic
from rl_power.modules.bus_attention_model import BusAttentionActor, BusAttentionCritic
from rl_power.training.a2c_trainer import A2CBranchTrainer

import torch
from torch import nn
import torch.nn.functional as F


def run_experiment(a, b, c, d, e, f, g, h, i, j, k, l):
    trainer = A2CBranchTrainer(env_sampling_config=a,
                               actor_type=b,
                               critic_type=c,
                               n_actions=d,
                               training_env=e,
                               model_linear_dim=f,
                               max_actions=g,
                               n_agents=h,
                               device=i,
                               batch_size=j,
                               entropy_coeff=k,
                               lr=l)

    trainer.train(25_000)
    # trainer.train(200)
    trainer.plot_history(f"- Trained on {h} Agents", )
    test_case = "ieee_data/pglib_opf_case118_ieee.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=1, max_actions=g,
                                         limit=100)


def variance_f1(t):
    return 0.0


def variance_f2(t):
    return 0.01


def variance_f3(t):
    return 0.05


def variance_f4(t):
    return 0.1


if __name__ == '__main__':
    network_library_path_list = [os.path.abspath("ieee_data/WB5.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case14_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case57_ieee.m")]

    variance_list = [variance_f1, variance_f2, variance_f3, variance_f4]
    weight_list = [#[1, 0, 0, 0],
                   [1, 1, 1, 1],
                   [0, 1, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 1],
                   ]

    batch_sizes = [16, 64]
    model_linear_dims = [128, 256]
    entropies = [0.0, 0.001]

    max_actions = [5]
    n_actions = [5]
    n_train_agents = [3]
    lrs = [1e-4]

    sampler_options_list = [{"paths": network_library_path_list,
                             "weights": weight,
                             "gen_cost_mean_var": [0, variance],
                             "load_mean_var": [0, variance]}
                            for weight, variance in itertools.product(weight_list, variance_list)]

    actor_type = [MLPCombinedActor]
    critic_type = [MLPCombinedCritic]
    training_env = [SampledEdgeEnv]
    device = ["cuda:0"]

    args_list = list(itertools.product(sampler_options_list,
                                       actor_type,
                                       critic_type,
                                       n_actions,
                                       training_env,
                                       model_linear_dims,
                                       max_actions,
                                       n_train_agents,
                                       device,
                                       batch_sizes,
                                       entropies,
                                       lrs))

    with Pool(processes=4) as pool:
        results = pool.starmap(run_experiment, args_list)

    print("")
