import torch
import torch.nn as nn
import numpy as np


class RealNVP(nn.Module):
    def __init__(self, data_dim, n_c_layers=8, n_hidden=100, hidden_dims=2):
        super(RealNVP, self).__init__()
        assert (
            n_c_layers > 1
        ), "Need more than one coupling layer to transform all dimensions of x."

        self.base = torch.distributions.MultivariateNormal(
            torch.zeros(data_dim), torch.eye(data_dim)
        )

        self.mask = torch.arange(data_dim) % 2
        self.coupling_layers = []
        for _ in range(n_c_layers):
            self.coupling_layers.append(
                CouplingLayer(data_dim, self.mask, n_hidden=100, hidden_dims=2)
            )
            self.coupling_layers.append(BatchNorm(data_dim))
            self.mask = 1 - self.mask

        self.model = nn.Sequential(*self.coupling_layers)

    def forward(self, x):
        sum_log_det_J = x.new_zeros(x.size(0))

        for i in range(len(self.coupling_layers)):
            x, log_det_J = self.coupling_layers[i](x)
            sum_log_det_J += log_det_J

        return x, sum_log_det_J

    def reverse(self, x):
        sum_log_det_J = x.new_zeros(x.size(0))
        for i in reversed(range(len(self.coupling_layers))):
            coupling_layer = self.coupling_layers[i]
            x, log_det_J = coupling_layer.reverse(x)
            sum_log_det_J += log_det_J

        return x, sum_log_det_J


class CouplingLayer(nn.Module):
    def __init__(self, data_dim, mask, n_hidden=100, hidden_dims=2):
        super(CouplingLayer, self).__init__()
        self.mask = mask
        self.scale = ScaleTranslate(data_dim, n_hidden, hidden_dims)
        self.translate = ScaleTranslate(data_dim, n_hidden, hidden_dims)

    def forward(self, x):
        x_m = x * self.mask

        s = self.scale(x_m)
        t = self.translate(x_m)
        s = s * (1 - self.mask)
        t = t * (1 - self.mask)

        z = x_m + (1.0 - self.mask) * x * torch.exp(s) + t
        log_det_J = torch.sum(s, 1)
        return z, log_det_J

    def reverse(self, x):
        x_m = x * self.mask
        s = self.scale(x_m)
        t = self.translate(x_m)
        z = x_m + (1.0 - self.mask) * (x - t) * torch.exp(-s)
        log_det_J = -torch.sum(s, 1)
        return z, log_det_J


class ScaleTranslate(nn.Module):
    def __init__(self, data_dim, n_hidden=256, hidden_dims=2):
        super(ScaleTranslate, self).__init__()
        self.layers = []

        self.layers.append(nn.Linear(data_dim, n_hidden))
        self.layers.append(nn.ReLU())
        for _ in range(1, hidden_dims):
            self.layers.append(nn.Linear(n_hidden, n_hidden))
            self.layers.append(nn.ReLU())
        self.layers.append(nn.Linear(n_hidden, data_dim))

        self.model = nn.Sequential(*self.layers)

    def forward(self, x):
        return self.model(x)


class BatchNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.zeros(1, dim).normal_(0, 0.05))
        self.beta = nn.Parameter(torch.zeros(1, dim).normal_(0, 0.05))
        self.running_mean = torch.zeros(1, dim)
        self.running_var = torch.ones(1, dim)

    def forward(self, x):
        m = x.mean(dim=0)
        v = torch.mean((x - m) ** 2, axis=0) + self.eps  # x.var(dim=0) + self.eps
        self.running_mean = m
        self.running_var = v
        x_hat = (x - m) / torch.sqrt(v)
        x_hat = x_hat * torch.exp(self.gamma) + self.beta
        log_det = torch.sum(self.gamma) - 0.5 * torch.sum(torch.log(v))
        return x_hat, log_det

    def reverse(self, x):
        m = self.running_mean
        v = self.running_var

        x_hat = (x - self.beta) * torch.exp(-self.gamma) * torch.sqrt(v) + m
        log_det = torch.sum(-self.gamma + 0.5 * torch.log(v))
        return x_hat, log_det


class NegLogLik(nn.Module):
    def __init__(self, model):
        super(NegLogLik, self).__init__()
        self.base = model.base

    def __call__(self, z, sum_log_det_J):
        log_p = self.base.log_prob(z)
        return -(log_p + sum_log_det_J).mean()
