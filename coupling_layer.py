import torch
import torch.nn as nn

class CouplingLayer(nn.Module):
    
    def __init__(self, data_dim,mask_type):
        super(CouplingLayer, self).__init__()
        self.scale_translate = ScaleTranslate(data_dim)
        
        # reverse mask for every second coupling layer
        if mask_type % 2 ==0:
            self.mask = torch.tensor([[0.0, 1.0]])
        else:
            self.mask = torch.tensor([[1.0, 0.0]])
        
    def forward(self, x, reverse=False):
        if not reverse:
            mask = self.mask
            
            x_m = x * mask
            s,t = self.scale_translate(x_m)
            s = s * (1-mask)
            t = t * (1-mask)

            z = x_m + (1.0-mask)*(x*torch.exp(s) + t)
            log_det_J = torch.sum(s, 1)
            return z, log_det_J
        
        else:
            mask = self.mask

            x_m = x * mask
            s,t = self.scale_translate(x_m)

            z = x_m + (1.0-mask) * ((x-t) * torch.exp(-s))
            return z

class ScaleTranslate(nn.Module):
    
    def __init__(self, data_dim, n_hidden=256): 
        super(ScaleTranslate, self).__init__()
        
        layers = []
        layers.append(nn.Linear(data_dim, n_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, n_hidden))
        layers.append(nn.ReLU())
        layers.append(nn.Linear(n_hidden, data_dim))
        
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        t = self.model(x)
        s = torch.tanh(t)
        return s, t