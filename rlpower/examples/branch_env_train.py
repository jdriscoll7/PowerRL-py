from matplotlib import pyplot as plt
from ray.rllib.algorithms import DQNConfig, DQN
from ray.rllib.algorithms.algorithm import Algorithm
import ray
from ray import air
from ray import tune
import os

from rlpower.learning.power_env import BranchEnv
from rlpower.learning.power_eval import run_policy_on_branch_env
from rlpower.learning.visualization import plot_training_curve

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
                "fcnet_hiddens": [256, 256, 256],
                "fcnet_activation": "tanh",
                # "epsilon": [(0, 1.0), (10000, 0.02)],
                "fcnet_bias_initializer": "zeros_",
                "post_fcnet_bias_initializer": "zeros_",
                "post_fcnet_hiddens": [256],
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
            n_step=3,
            double_q=True,
            num_atoms=8,
            noisy=False,
            dueling=True,
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

    directory = os.path.abspath("./training_results")

    result = tune.run(DQN,
                      config=config,
                      stop={"training_iteration": 20},
                      local_dir=directory,
                      name="branch_experiment",
                      verbose=1,
                      checkpoint_at_end=True)

    checkpoint = result.get_last_checkpoint(result.get_best_trial())
    algo = Algorithm.from_checkpoint(checkpoint)

    results_path = result.get_best_trial().path
    plot_training_curve(results_path)
    plt.show()
    eval_env_options = {"path": os.path.abspath("ieee_data/pglib_opf_case30_ieee.m"), "render": True}
    eval_env = BranchEnv(eval_env_options)
    run_policy_on_branch_env(eval_env, algo, 5)

    print("")
