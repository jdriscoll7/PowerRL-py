from torch import nn
import torch.nn.functional as F
from gymnasium.spaces import Space


class BusAttentionCritic(nn.Module):
    def __init__(self, state_length: int, linear_dim: int = 32, attention_embed_dim: int = 16, device: str = "cpu", n_heads: int = 4):
        super().__init__()
        self.device = device
        self.state_length = state_length
        self.linear_dim = linear_dim

        self.embedding = nn.Linear(in_features=state_length, out_features=attention_embed_dim, device=self.device)
        self.attention_layer = nn.MultiheadAttention(embed_dim=attention_embed_dim, num_heads=n_heads, batch_first=True)

        self.head = nn.Sequential(nn.Linear(in_features=attention_embed_dim, out_features=linear_dim, device=self.device),
                                  nn.ReLU(),
                                  nn.Linear(in_features=linear_dim, out_features=1, device=self.device))

    def forward(self, x):
        x = x.sign() * x.abs().pow(1 / 7)

        x = self.embedding(x)
        x, _ = self.attention_layer(x, x, x)
        x = self.head(x)

        return x


class BusAttentionActor(nn.Module):
    def __init__(self, state_length: int, linear_dim: int = 32, attention_embed_dim: int = 16, device: str = "cpu", n_heads: int = 4,
                 n_actions: int = 3):
        super().__init__()
        self.device = device
        self.state_length = state_length
        self.linear_dim = linear_dim

        self.embedding = nn.Linear(in_features=state_length, out_features=attention_embed_dim, device=self.device)
        self.attention_layer = nn.MultiheadAttention(embed_dim=attention_embed_dim, num_heads=n_heads, batch_first=True)

        self.head = nn.Sequential(nn.Linear(in_features=attention_embed_dim, out_features=linear_dim, device=self.device),
                                  nn.LeakyReLU(negative_slope=0.02),
                                  nn.Linear(in_features=linear_dim, out_features=n_actions, device=self.device))

    def forward(self, x):
        x = x.sign() * x.abs().pow(1 / 7)

        x = self.embedding(x)
        x, _ = self.attention_layer(x, x, x)
        x = self.head(x)

        x = F.softmax(x, dim=-1)

        return x
