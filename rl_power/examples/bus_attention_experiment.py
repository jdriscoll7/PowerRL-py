import os

from rl_power.modules.bus_attention_model import BusAttentionActor, BusAttentionCritic
from rl_power.training.a2c_trainer import A2CBranchTrainer

if __name__ == '__main__':
    variance = 1
   # sampler_options = {"paths": [os.path.abspath("ieee_data/WB5.m"),
    #                              os.path.abspath("ieee_data/pglib_opf_case14_ieee.m"),
    #                              os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"),
    #                              os.path.abspath("ieee_data/pglib_opf_case57_ieee.m")],
    #                    "weights": [0.1, 0.4, 0.3, 0.2]
    #                    }

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
    trainer = A2CBranchTrainer(actor_type=BusAttentionActor,
                               critic_type=BusAttentionCritic,
                               env_sampling_config=sampler_options,
                               model_linear_dim=32,
                               model_attn_dim=128,
                               n_heads=2,
                               max_actions=max_actions,
                               n_agents=n_train_agents,
                               device="cuda:0",
                               batch_size=64,
                               entropy_coeff=1e-4,
                               lr=1e-4,)

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