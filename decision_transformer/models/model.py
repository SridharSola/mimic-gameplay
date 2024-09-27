import numpy as np
import torch
import torch.nn as nn


class TrajectoryModel(nn.Module):

    def __init__(self, state_dim, mouse_dim, key_dim, max_length=None):
        super().__init__()

        self.state_dim = state_dim
        self.mouse_dim = mouse_dim
        self.key_dim = key_dim
        self.max_length = max_length

    def forward(self, states, mouse_actions, key_actions, masks=None, attention_mask=None):
        # "masked" tokens or unspecified inputs can be passed in as None
        return None, None, None

    def get_action(self, states, mouse_actions, key_actions, **kwargs):
        # these will come as tensors on the correct device
        return torch.zeros_like(mouse_actions[-1]), torch.zeros_like(key_actions[-1])