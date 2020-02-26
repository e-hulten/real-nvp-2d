import torch
import torch.nn as nn
import numpy as np
from made import MADE
from batchnorm import BatchNorm


class NewFlow(nn.Module):
    def __init__(self, data_dim, n_c_layers=8, n_hidden=512, hidden_dims=2):
        super(NewFlow, self).__init__()

        self.mask = torch.arange(data_dim) % 2
        self.layers = nn.ModuleList()
        self.made_rev = False

        l = 0
        for _ in range(n_c_layers):
            self.layers.append(
                NewCouplingLayer(
                    data_dim=data_dim,
                    mask=self.mask,
                    n_hidden=n_hidden,
                    hidden_dims=hidden_dims,
                    made_rev=self.made_rev,
                )
            )
            self.layers.append(BatchNorm(data_dim))
            self.mask = 1 - self.mask

            if l == 2:
                self.made_rev = not self.made_rev
                l = 0
            else:
                l += 1

            # l += 1
            # if l % 2 == 0:
            #    self.mask = 1 - self.mask

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
        for layer in reversed(self.layers):
            x, log_det_J = layer.reverse(x)
            sum_log_det_J += log_det_J

        return x, sum_log_det_J


class NewCouplingLayer(nn.Module):
    def __init__(self, data_dim, mask, hidden_dims, n_hidden, made_rev):
        super(NewCouplingLayer, self).__init__()
        self.mask = mask == 1
        self.made_rev = made_rev

        self.n_1 = np.ceil(data_dim / 2).astype(int)
        self.n_2 = np.floor(data_dim / 2).astype(int)

        self.made = MADE(n_in=self.n_1, hidden_dims=[n_hidden], gaussian=True)
        self.scale = ScaleTranslate(
            self.n_1, self.n_2, n_hidden, hidden_dims, actfun="tanh"
        )
        self.translate = ScaleTranslate(self.n_1, self.n_2, n_hidden, hidden_dims)

    def forward(self, x):
        u = torch.zeros_like(x)
        x_1 = x[:, self.mask]
        x_2 = x[:, ~self.mask]

        # real nvp
        s = self.scale(x_1)
        t = self.translate(x_1)
        u_rnvp = (x_2 - t) * torch.exp(-s)

        # made
        if not self.made_rev:
            out = self.made(x_1.float())
            mu, alpha = torch.chunk(out, 2, dim=1)
            u_made = (x_1 - mu) * (torch.exp(-alpha) + 1e-5)
        else:
            x_1 = x_1.flip(dims=(1,))
            out = self.made(x_1.float())
            mu, alpha = torch.chunk(out, 2, dim=1)
            u_made = (x_1 - mu) * (torch.exp(-alpha) + 1e-5)
            u_made = u_made.flip(dims=(1,))

        # fill in u
        u[:, self.mask] = u_made
        u[:, ~self.mask] = u_rnvp

        log_det_J = -torch.sum(s, dim=1) - torch.sum(alpha, dim=1)
        return u, log_det_J

    def reverse(self, u):
        u = u.flip(dims=(1,))
        x = torch.zeros_like(u)
        u_1 = u[:, self.mask]
        u_2 = u[:, ~self.mask]

        # inverse made
        x_made = torch.zeros_like(u_1)
        for dim in range(self.n_1):
            out = self.made(x_made)
            mu, alpha = torch.chunk(out, 2, dim=1)
            x_made[:, dim] = mu[:, dim] + u_1[:, dim] * (
                torch.exp(alpha[:, dim]) + 1e-5
            )

        # inverse real nvp
        s = self.scale(x_made)
        t = self.translate(x_made)
        x_rnvp = u_2 * torch.exp(s) + t

        # fill in
        x[:, ~self.mask] = x_rnvp
        x[:, self.mask] = x_made
        log_det_J = torch.sum(s, dim=1) + torch.sum(alpha, dim=1)
        return x, log_det_J


class ScaleTranslate(nn.Module):
    def __init__(self, in_dim, out_dim, n_hidden, hidden_dims, actfun="relu"):
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
