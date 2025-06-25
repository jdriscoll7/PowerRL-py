import random
from collections import deque, namedtuple

import torch

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

LSTMTransition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward', 'lstm_state', 'lstm_cell'))


class ReplayMemory(object):

    def __init__(self, capacity, use_lstm: bool = False):
        self.memory = deque([], maxlen=capacity)
        self.use_lstm = use_lstm

    def push(self, state, action, next_state, reward, lstm_state=None, device='cuda'):

        assert self.use_lstm and (lstm_state is not None) or not self.use_lstm

        if self.use_lstm:
            for k in state.keys():
                self.memory.append(LSTMTransition(state[k],
                                                  torch.tensor([action[k]], device=device),
                                                  next_state[k],
                                                  reward[k],
                                                  lstm_state[0][k],
                                                  lstm_state[1][k]))

        else:
            for k in state.keys():
                self.memory.append(Transition(state[k],
                                              torch.tensor([action[k]], device=device),
                                              next_state[k],
                                              reward[k]))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)