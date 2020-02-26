import torch
import torch.nn as nn
import numpy as np
from batchnorm import BatchNorm


class RealNVP(nn.Module):
    def __init__(self, data_dim, n_c_layers=8, n_hidden=100, hidden_dims=2):
        super(RealNVP, self).__init__()
        mask = torch.arange(data_dim) % 2

        self.layers = nn.ModuleList()
        for _ in range(n_c_layers):
            self.layers.append(
                CouplingLayer(
                    data_dim, mask, n_hidden=n_hidden, hidden_dims=hidden_dims
                )
            )
            self.layers.append(BatchNorm(data_dim))
            mask = 1 - mask

    def forward(self, x, set_batch_stats=False):
        sum_log_det_J = x.new_zeros(x.size(0))

        for layer in self.layers:
            if isinstance(layer, BatchNorm):
                x, log_det_J = layer(x, set_batch_stats=set_batch_stats)
            else:
                x, log_det_J = layer(x)
            sum_log_det_J += log_det_J

        return x, sum_log_det_J

    def reverse(self, x):
        sum_log_det_J = x.new_zeros(x.size(0))
        for layer in reversed((self.layers)):
            x, log_det_J = layer.reverse(x)
            sum_log_det_J += log_det_J

        return x, sum_log_det_J


class CouplingLayer(nn.Module):
    def __init__(self, data_dim, mask, n_hidden, hidden_dims):
        super(CouplingLayer, self).__init__()
        self.mask = mask
        self.scale = ScaleTranslate(data_dim, n_hidden, hidden_dims, actfun="tanh")
        self.translate = ScaleTranslate(data_dim, n_hidden, hidden_dims)

    def forward(self, x):
        x_m = x * self.mask
        s = self.scale(x_m)
        t = self.translate(x_m)
        u = x_m + (1.0 - self.mask) * (x - t) * torch.exp(-s)
        log_det_J = -torch.sum((1 - self.mask) * s, 1)
        return u, log_det_J

    def reverse(self, x):
        x_m = x * self.mask
        s = self.scale(x_m)
        t = self.translate(x_m)
        z = x_m + (1.0 - self.mask) * (x * torch.exp(s) + t)
        log_det_J = torch.sum((1 - self.mask) * s, 1)
        return z, log_det_J


class ScaleTranslate(nn.Module):
    def __init__(self, data_dim, n_hidden, hidden_dims, actfun="relu"):
        super(ScaleTranslate, self).__init__()

        if actfun == "relu":
            self.actfun = nn.ReLU()
        elif actfun == "tanh":
            self.actfun = nn.Tanh()

        self.layers = []
        self.layers.append(nn.Linear(data_dim, n_hidden))
        self.layers.append(self.actfun)
        for _ in range(1, hidden_dims):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(self.actfun)
        self.layers.append(nn.Linear(n_hidden, data_dim))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)

