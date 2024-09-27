from matplotlib import pyplot as plt
from ray.rllib.algorithms import DQNConfig, DQN
from ray.rllib.algorithms.algorithm import Algorithm
import os
from stable_baselines3 import DQN

from rl_power.envs.branch_env import BranchEnv
from rl_power.training.power_eval import run_policy_on_branch_env
from rl_power.visualization.visualization import plot_training_curve


class BranchEnvTrainer:

    def __init__(self, env: BranchEnv):

        self.model = DQN("MlpPolicy", env, verbose=1)
        self.training_rounds = 1

        env_options = {"path": os.path.abspath("ieee_data/pglib_opf_case30_ieee.m")}

    def train(self, n_iterations: int = 1000):

        self.model.learn(total_timesteps=n_iterations, log_interval=100)
        self.model.save(f"trained_models/branch_env_model_{self.training_rounds}")

        self.training_rounds += 1

    def test(self, env: BranchEnv = None):

        obs, info = env.reset()

        for i in range(5):
            action, _states = self.model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            if terminated or truncated:
                obs, info = env.reset()



if __name__ == '__main__':

    env_options = {"path": os.path.abspath("ieee_data/pglib_opf_case30_ieee.m")}
    env = BranchEnv(env_options)

    trainer = BranchEnvTrainer(env=env)
    trainer.train(10000)

    # run_policy_on_branch_env(eval_env, algo, 5)
