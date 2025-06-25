from rl_power.envs.edge_agent_branch_env import EdgeAgentBranchEnv

import os
import torch
from torch import nn
import torch.nn.functional as F
import matplotlib.pyplot as plt

from rl_power.power.drawing import PMSolutionRenderer

if __name__ == '__main__':
    plt.show()

    env = EdgeAgentBranchEnv(render_mode=True,
                             path=os.path.abspath("ieee_data/WB5.m"),
                             n_agents=1)

    env.agents = ["3"]
    agents = env.agents

    action = {agent: 0 for agent in agents}
    env.step(action)

    action[agents[0]] = 4
    env.step(action)

    print("")