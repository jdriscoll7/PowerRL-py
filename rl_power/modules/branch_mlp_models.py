import torch
from torch import nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler


class MinMaxPreprocessor(nn.Module):
    def __init__(self, target_range=(-1, 1)):
        super(MinMaxPreprocessor, self).__init__()
        self.running_maxs = None
        self.running_mins = None
        self.target_range = target_range

    def forward(self, x):
        # Min and max over both the time-feature dimension and the batch dimension.
        mins, maxs = torch.min(x, dim=-2, keepdim=True).values, torch.max(x, dim=-2, keepdim=True).values
        mins, maxs = torch.min(mins, dim=0, keepdim=True).values, torch.max(maxs, dim=0, keepdim=True).values

        # Prevent issues with zero-mins and zero-maxs by converting zeros to ones.
        mins[mins == 0.0] = -1
        maxs[maxs == 0.0] = 1
        maxs[maxs == mins] += 1

        if self.running_mins is None or self.running_maxs is None:
            self.running_mins = mins
            self.running_maxs = maxs
        else:
            self.running_mins = torch.min(self.running_mins, mins)
            self.running_maxs = torch.max(self.running_maxs, maxs)

        x = (x - self.running_mins) / (self.running_maxs - self.running_mins) * 2 - 1

        return x


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
        x = x.sign() * x.abs().pow(1 / 7)

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
        x = x.sign() * x.abs().pow(1 / 7)

        x = self.layer_1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_3(x)

        return x


class MLPCombinedActor(nn.Module):
    def __init__(self, state_length: int, n_agents: int, linear_dim: int = 256, n_actions: int = 8, device: str = "cpu",
                 **kwargs):
        super().__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.state_length = state_length
        self.linear_dim = linear_dim
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.layer_1 = nn.Linear(in_features=state_length * n_agents, out_features=linear_dim, device=self.device)
        self.layer_2 = nn.Linear(in_features=linear_dim, out_features=linear_dim, device=self.device)
        self.layer_3 = nn.Linear(in_features=linear_dim, out_features=n_agents * n_actions, device=self.device)

        self.preprocessor = MinMaxPreprocessor(target_range=(-1, 1))

    def forward(self, x):
        # x = x.sign() * x.abs().pow(1 / 7)

        x = self.preprocessor(x)

        x = x.flatten(start_dim=1)

        x = self.layer_1(x)
        x = nn.functional.elu(x)
        x = self.layer_2(x)
        x = nn.functional.elu(x)
        x = self.layer_3(x)

        x = torch.reshape(x, (-1, self.n_agents, self.n_actions))

        return F.softmax(x, dim=-1)


class MLPCombinedCritic(nn.Module):
    def __init__(self, state_length: int, n_agents: int, linear_dim: int = 256, n_actions: int = 8, device: str = "cpu",
                 **kwargs):
        super().__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.state_length = state_length
        self.linear_dim = linear_dim
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.layer_1 = nn.Linear(in_features=state_length * n_agents, out_features=linear_dim, device=self.device)
        self.layer_2 = nn.Linear(in_features=linear_dim, out_features=linear_dim, device=self.device)
        self.layer_3 = nn.Linear(in_features=linear_dim, out_features=1, device=self.device)
        self.preprocessor = MinMaxPreprocessor(target_range=(-1, 1))

    def forward(self, x):
        # x = x.sign() * x.abs().pow(1 / 7)

        x = self.preprocessor(x)

        x = x.flatten(start_dim=1)

        x = self.layer_1(x)
        x = nn.functional.elu(x)
        x = self.layer_2(x)
        x = nn.functional.elu(x)
        x = self.layer_3(x)

        return x
