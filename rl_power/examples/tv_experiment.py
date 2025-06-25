import os

from rl_power.envs.time_varying_branch_env import NetworkValueController, TVBranchEnv
from rl_power.examples.bus_mlp_experiment import MLPActor, MLPCritic
from rl_power.training.a2c_tv_trainer import A2CTVBranchTrainer

if __name__ == '__main__':
    # path = os.path.abspath("ieee_data/pglib_opf_case57_ieee.m")
    path = os.path.abspath("ieee_data/WB5.m")
    # path = os.path.abspath("ieee_data/pglib_opf_case30_ieee.m")

    # load_from_memory = True
    load_from_memory = False
    max_actions = 5
    n_train_agents = 2

    net_controller = NetworkValueController(gen_cost_mean_var=[0, 1], load_mean_var=[0, 1])
    tv_env = TVBranchEnv(path=path, agents=[1, 2, 3], max_actions=10, network_controller=net_controller, )

    trainer = A2CTVBranchTrainer(actor_type=MLPActor,
                                 critic_type=MLPCritic,
                                 tv_environment=tv_env,
                                 model_linear_dim=64,
                                 max_actions=max_actions,
                                 n_agents=n_train_agents,
                                 device="cuda:0",
                                 batch_size=64,
                                 entropy_coeff=1e-4,
                                 lr=1e-3)

    if load_from_memory:
        trainer.load_latest_model()
        results = None
    else:
        results = trainer.train(10000)

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
