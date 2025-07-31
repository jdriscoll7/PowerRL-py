import torch
from torch import nn
import torch.nn.functional as F

from sklearn.preprocessing import MinMaxScaler

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
    def __init__(self, state_length: int, n_agents: int, linear_dim: int = 256, n_actions: int = 8, device: str = "cpu", **kwargs):
        super().__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.state_length = state_length
        self.linear_dim = linear_dim
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.layer_1 = nn.Linear(in_features=state_length*n_agents, out_features=linear_dim, device=self.device)
        self.layer_2 = nn.Linear(in_features=linear_dim, out_features=linear_dim, device=self.device)
        self.layer_3 = nn.Linear(in_features=linear_dim, out_features=n_agents*n_actions, device=self.device)

        self.preprocessor = MinMaxScaler(feature_range=(-1, 1))


    def forward(self, x):
        # x = x.sign() * x.abs().pow(1 / 7)

        # x_original_shape = x.shape
        # temp_shape = (x.shape[0], -1)
        #
        # self.preprocessor.partial_fit(x.view(temp_shape).cpu().numpy())
        # x = self.preprocessor.transform(x.view(temp_shape).cpu())
        #
        # x = torch.tensor(x.reshape(x_original_shape), device=self.device, dtype=torch.float)

        x = torch.sign(x) * torch.log(torch.abs(x) + 1)

        x = x.flatten()

        x = self.layer_1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_3(x)

        x = torch.reshape(x, (self.n_agents, 1, self.n_actions))

        return F.softmax(x, dim=-1)


class MLPCombinedCritic(nn.Module):
    def __init__(self, state_length: int, n_agents: int, linear_dim: int = 256, n_actions: int = 8, device: str = "cpu", **kwargs):
        super().__init__()
        # self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        self.device = device

        self.state_length = state_length
        self.linear_dim = linear_dim
        self.n_agents = n_agents
        self.n_actions = n_actions

        self.layer_1 = nn.Linear(in_features=state_length*n_agents, out_features=linear_dim, device=self.device)
        self.layer_2 = nn.Linear(in_features=linear_dim, out_features=linear_dim, device=self.device)
        self.layer_3 = nn.Linear(in_features=linear_dim, out_features=1, device=self.device)
        self.preprocessor = MinMaxScaler(feature_range=(-1, 1))

    def forward(self, x):
        # x = x.sign() * x.abs().pow(1 / 7)

        # x_original_shape = x.shape
        # temp_shape = (x.shape[0], -1)
        #
        # self.preprocessor.partial_fit(x.view(temp_shape).cpu().numpy())
        # x = self.preprocessor.transform(x.view(temp_shape).cpu())
        #
        # x = torch.tensor(x.reshape(x_original_shape), device=self.device, dtype=torch.float)

        x = torch.sign(x) * torch.log(torch.abs(x) + 1)
        x = x.flatten()

        x = self.layer_1(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_2(x)
        x = nn.functional.leaky_relu(x, negative_slope=0.02)
        x = self.layer_3(x)

        return x