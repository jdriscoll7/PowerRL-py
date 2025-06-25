import torch
from torch import nn


class DefaultNetwork(nn.Module):
    def __init__(self, state_length: int, linear_dim: int = 256, n_actions: int = 8, use_lstm: bool = False, **kwargs):
        super().__init__()
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"

        self.use_lstm = use_lstm
        self.state_length = state_length
        self.linear_dim = linear_dim

        self.layer_1 = nn.Linear(in_features=state_length, out_features=linear_dim, device=self.device)

        if use_lstm:
            self.lstm_layer = nn.LSTM(input_size=linear_dim, hidden_size=linear_dim, device=self.device, batch_first=True)

        self.layer_2 = nn.Linear(in_features=linear_dim, out_features=linear_dim, device=self.device)
        self.layer_3 = nn.Linear(in_features=linear_dim, out_features=n_actions, device=self.device)

    def forward(self, x, hidden=None):

        x = x.sign() * x.abs().pow(1/5)

        x = self.layer_1(x)
        x = nn.functional.relu(x)

        if self.use_lstm:
            x = x.reshape(-1, 1, self.linear_dim)
            if hidden is None:
                hidden = self.get_hidden(x)

            x, new_hidden = self.lstm_layer(x, hidden)
            x = x.reshape(-1, self.linear_dim)

        x = self.layer_2(x)
        x = nn.functional.relu(x)
        x = self.layer_3(x)

        if self.use_lstm:
            return x, new_hidden
        else:
            return x

    def get_hidden(self, x):

        hidden = (
                torch.zeros(1, x.shape[0], self.linear_dim, device=x.device),
                torch.zeros(1, x.shape[0], self.linear_dim, device=x.device),
                )
        return hidden
