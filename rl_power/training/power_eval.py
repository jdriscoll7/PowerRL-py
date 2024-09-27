from rl_power.envs.branch_env import BranchEnv
from ray.rllib.algorithms.algorithm import Algorithm
import matplotlib.pyplot as plt


def run_policy_on_branch_env(env: BranchEnv, algo: Algorithm, n_actions: int, path: str = None):

    plt.show()
    episode_reward = 0
    action_count = 0
    # done = False
    # obs, info = env.reset()
    obs = env.get_observation()
    while action_count < n_actions:
        action = algo.compute_single_action(obs)
        obs, reward, done, truncated, info = env.step(action)
        episode_reward += reward
        action_count += 1

    return episode_reward
