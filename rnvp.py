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
    def __init__(self, data_dim, mask, n_hidden=100, hidden_dims=2):
        super(CouplingLayer, self).__init__()
        self.mask = mask % 2 == 0  # make mask boolean

        # number of elements in each partition of the coupling layer
        n_1 = self.mask.sum().numpy()
        n_2 = (~self.mask).sum().numpy()

        self.scale = ScaleTranslate(n_1, n_2, n_hidden, hidden_dims, actfun="tanh")
        self.translate = ScaleTranslate(n_1, n_2, n_hidden, hidden_dims)

    def forward(self, x):
        x_1 = x[:, self.mask]
        x_2 = x[:, ~self.mask]

        # real nvp
        s = self.scale(x_1)
        t = self.translate(x_1)
        x_2 = (x_2 - t) * torch.exp(-s)

        # fill in
        x[:, ~self.mask] = x_2

        log_det_J = -torch.sum(s, 1)
        return x, log_det_J

    def reverse(self, x):
        x_new = torch.zeros_like(x)

        x_1 = x[:, self.mask]
        x_2 = x[:, ~self.mask]

        s = self.scale(x_1)
        t = self.translate(x_1)
        x_2 = x_2 * torch.exp(s) + t

        # fill in
        x_new[:, self.mask] = x_1
        x_new[:, ~self.mask] = x_2
        log_det_J = torch.sum(s, 1)
        return x_new, log_det_J


class ScaleTranslate(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden=256, hidden_dims=2, actfun="relu"):
        super(ScaleTranslate, self).__init__()
        if actfun == "relu":
            self.actfun = nn.ReLU()
        elif actfun == "tanh":
            self.actfun = nn.Tanh()

        self.layers = []
        self.layers.append(nn.Linear(in_dim, n_hidden))
        self.layers.append(self.actfun)
        for _ in range(1, hidden_dims):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(self.actfun)
        self.layers.append(nn.Linear(n_hidden, out_dim))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)
