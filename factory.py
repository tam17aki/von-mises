# -*- coding: utf-8 -*-
"""A Python module which provides optimizer, scheduler, and customized loss.

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
from omegaconf import DictConfig
from torch import nn, optim


def get_optimizer(cfg: DictConfig, model):
    """Instantiate optimizer.

    Args:
        cfg (DictConfig): configuration in YAML format.
        model (nn.Module): network parameters.
    """
    optimizer_class = getattr(optim, cfg.training.optim.optimizer.name)
    optimizer = optimizer_class(
        model.parameters(), **cfg.training.optim.optimizer.params
    )
    return optimizer


def get_lr_scheduler(cfg: DictConfig, optimizer):
    """Instantiate scheduler.

    Args:
        cfg (DictConfig): configuration in YAML format.
    """
    lr_scheduler_class = getattr(
        optim.lr_scheduler, cfg.training.optim.lr_scheduler.name
    )
    lr_scheduler = lr_scheduler_class(
        optimizer, **cfg.training.optim.lr_scheduler.params
    )
    return lr_scheduler


class CustomLoss(nn.Module):
    """Custom loss."""

    def __init__(self, model, device):
        """Initialize class."""
        super().__init__()
        self.model = model
        self.device = device

    def _compute_grd_loss(self, predicted, reference):
        """Compute group delay loss.

        Args:
            predicted: estimated phase via von Mises DNN [N, T, C].
            reference: ground-truth phase [N, T, C].

        Returns:
            cosine loss of group delay.
        """
        predicted_grd = -predicted[:, :, 1:] + predicted[:, :, :-1]
        reference_grd = -reference[:, :, 1:] + reference[:, :, :-1]
        diff = predicted_grd - reference_grd
        loss = torch.sum(-torch.cos(diff), dim=-1)  # sum along frequency axis
        loss = torch.sum(loss, dim=-1)  # sum along time axis
        return loss.mean()  # average along batch axis

    def forward(self, batch):
        """Compute loss.

        Args:
            batch (Tuple): tuple of minibatch.

        Returns:
            loss: cosine loss of phase.
            grd_loss: cosine loss of group delay.
        """
        logabs_batch, phase_batch = batch
        logabs_batch = logabs_batch.to(self.device).float()
        phase_batch = phase_batch.to(self.device).float()
        predicted = self.model(logabs_batch)
        loss = -torch.cos(predicted - phase_batch)
        loss = torch.sum(loss, dim=-1)  # sum along frequency axis
        loss = torch.sum(loss, dim=-1)  # sum along time axis
        loss = loss.mean()  # average along batch axis
        grd_loss = self._compute_grd_loss(predicted, phase_batch)
        return loss, grd_loss


def get_loss(model, device):
    """Instantiate customized loss."""
    custom_loss = CustomLoss(model, device)
    return custom_loss
