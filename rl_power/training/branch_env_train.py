from matplotlib import pyplot as plt
from ray.rllib.algorithms import DQNConfig, DQN
from ray.rllib.algorithms.algorithm import Algorithm
import ray
from ray import air
from ray import tune
import os

from rl_power.envs.branch_env import BranchEnv
from rl_power.training.power_eval import run_policy_on_branch_env
from rl_power.visualization.visualization import plot_training_curve

if __name__ == '__main__':

    ray.init()

    env_options = {"path": os.path.abspath("ieee_data/pglib_opf_case30_ieee.m")}

    config = (
        DQNConfig()
        .environment(env=BranchEnv, env_config=env_options)
        .resources(num_cpus_for_local_worker=1)
        .rl_module(
            # Settings identical to old stack.
            model_config_dict={
                "fcnet_hiddens": [256, 256],
                "fcnet_activation": "relu",
                # "epsilon": [(0, 1.0), (3000, 0.01)],
                # "fcnet_bias_initializer": "zeros_",
                # "post_fcnet_bias_initializer": "zeros_",
                # "post_fcnet_hiddens": [256],
                "use_attention": True,
                "attention_use_n_prev_actions": 5,
                "attention_use_n_prev_rewards": 5,
            },
        )
        .training(
            # Settings identical to old stack.
            # train_batch_size_per_learner=8,
            # replay_buffer_config={
            #     "type": "PrioritizedEpisodeReplayBuffer",
            #     "capacity": 50000,
            #     "alpha": 0.6,
            #     "beta": 0.4,
            # },
            n_step=5,
            double_q=False,
            num_atoms=1,
            noisy=False,
            dueling=False,
            target_network_update_freq=20,
        )
        # .evaluation(
        #     evaluation_interval=1,
        #     evaluation_parallel_to_training=True,
        #     evaluation_num_env_runners=1,
        #     evaluation_duration="auto",
        #     evaluation_config={
        #         "explore": False,
        #         "metrics_num_episodes_for_smoothing": 4,
        #     },
        # )
    )

    directory = os.path.abspath("./ray_results")
    tuner = tune.Tuner(DQN,
                       param_space=config,
                       run_config=air.RunConfig(stop={"training_iteration": 3},
                                                name="branch_experiment",
                                                verbose=1,
                                                checkpoint_config=air.CheckpointConfig(checkpoint_at_end=True),
                                                storage_path=directory)
                       )
    results = tuner.fit()

    algo = Algorithm.from_checkpoint(results.get_best_result().checkpoint)

    results_path = results.get_best_result().path
    plot_training_curve(results_path)
    plt.show()
    eval_env_options = {"path": os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"), "render": True}
    eval_env = BranchEnv(eval_env_options)
    run_policy_on_branch_env(eval_env, algo, 5)

    print("")
