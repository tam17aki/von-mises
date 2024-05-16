# -*- coding: utf-8 -*-
"""Model definition of von Mises DNN for phase reconstruction.

Copyright (C) 2024 by Akira TAMAMORI

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""

import torch
from torch import nn


class GLU(nn.Module):
    """Gated Linear Unit."""

    def __init__(self, input_dim, output_dim):
        """Initialize module."""
        super().__init__()
        self.linear1 = nn.Linear(input_dim, output_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.activation = nn.GELU()

    def forward(self, inputs):
        """Apply GLU."""
        return self.linear1(inputs) * self.activation(self.linear2(inputs))


class FFNN(nn.Module):
    """Feed-Forward Neural Network."""

    def __init__(self, config):
        """Initialize class."""
        super().__init__()
        input_dim = config.model.input_dim
        hidden_dim = config.model.hidden_dim
        win_width = 2 * config.model.win_range + 1
        net = nn.ModuleList([nn.Linear(input_dim * win_width, hidden_dim)])
        for _ in range(config.model.n_layers):
            net += [GLU(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim)]
        net += [nn.Linear(hidden_dim, input_dim)]
        self.net = nn.Sequential(*net)

    def forward(self, inputs):
        """Forward propagation."""
        return self.net(inputs)


def get_model(cfg, device: torch.device):
    """Instantiate models.

    Args:
        cfg: configuration of models
    """
    model = FFNN(cfg).to(device)
    return model
