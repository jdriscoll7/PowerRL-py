from ray.rllib.models import ModelCatalog
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from torch import nn
from gymnasium.spaces import Space


class BusAttentionModel(TorchModelV2):
    def __init__(self, obs_space: Space, action_space: Space, num_outputs: int, model_config: dict, name: str,
                 max_degree: int = 16):

        super().__init__(obs_space, action_space, num_outputs, model_config, name)

        # Store some dimensions to determine layer dimensions later.
        self.max_degree = max_degree
        self.observation_shape = obs_space.shape
        self.branch_config_size = action_space.shape[0]

        # Input embedding layer (add normalization?). Embed raw features for all adjacent branches.

        # Self-attention (replace with transformer?) across all adjacent branches. Add two special input tokens for
        # special stay/end inputs, as well as padding tokens up to self.max_degree.

        # Model has 3 output heads: (1) branch selection, (2) next bus selection, (3) branch configuration.

        # (1): Linear layer that flattens branch attention outputs, followed by softmax.

        # (2): Linear layer that flattens branch + 2 special token attention outputs, followed by softmax.

        # (3): Linear layer along feature dimension to convert to config size.

    def forward(self, input_dict, state, seq_lens):
        pass

    @staticmethod
    def register():
        ModelCatalog.register_custom_model("bus_attention_model", BusAttentionModel)
