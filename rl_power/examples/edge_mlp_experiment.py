import os

from rl_power.envs.edge_agent_branch_env import SampledEdgeEnv
from rl_power.envs.time_varying_branch_env import NetworkValueController
from rl_power.modules.branch_mlp_models import MLPCombinedActor, MLPCombinedCritic, MLPActor, MLPCritic
from rl_power.modules.bus_attention_model import BusAttentionActor, BusAttentionCritic
from rl_power.training.a2c_trainer import A2CBranchTrainer

import torch
from torch import nn
import torch.nn.functional as F



if __name__ == '__main__':
    # variance = lambda t: 0.01
    variance = lambda t: 0.0
    # variance = lambda t: t/50e3
    sampler_options = {"paths": [os.path.abspath("ieee_data/WB5.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case14_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
                                 os.path.abspath("ieee_data/pglib_opf_case57_ieee.m")],
                       # "weights": [3, 1, 0.33, 0.11],
                       "weights": [1, 0, 0, 0],
                       # "weights": [0, 0, 1, 0],
                       # "weights": [1, 1, 1, 1],
                       "gen_cost_mean_var": [0, variance],
                       "load_mean_var": [0, variance],
                       }

    use_lstm = False
    # load_from_memory = True
    load_from_memory = False
    max_actions = 5
    n_train_agents = 3

    print(f"LSTM = {use_lstm}")
    trainer = A2CBranchTrainer(env_sampling_config=sampler_options,
                               # actor_type=MLPActor,
                               # critic_type=MLPCritic,
                               actor_type=MLPCombinedActor,
                               critic_type=MLPCombinedCritic,
                               n_actions=5,
                               training_env=SampledEdgeEnv,
                               model_linear_dim=256,
                               max_actions=max_actions,
                               n_agents=n_train_agents,
                               device="cuda:0",
                               batch_size=1,
                               entropy_coeff=0,
                               lr=1e-4)

    if load_from_memory:
        trainer.load_latest_model()
        results = None
    else:
        results = trainer.train(50000)

    test_case = "ieee_data/WB5.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=3, max_actions=max_actions, limit=100)

    trainer.plot_history(f"- Trained on {n_train_agents} Agents")

    # test_case = "ieee_data/pglib_opf_case14_ieee.m"
    # trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=3)
    #
    # test_case = "ieee_data/pglib_opf_case30_ieee.m"
    # trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=3)

    test_case = "ieee_data/pglib_opf_case57_ieee.m"
    trainer.agent_count_evaluation_sweep(test_case=test_case, min_agents=1, max_agents=2, max_actions=max_actions, limit=100)

    # trainer.full_evaluation(), n_agents=2)

    # trainer.plot_history()
    # trainer.evaluate(os.path.abspath("ieee_data/pglib_opf_case57_ieee.m"), active_branches=['8'])

    print("")

    # run_policy_on_branch_env(eval_env, algo, 5)
