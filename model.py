import torch
import torch.nn as nn
from coupling_layer import CouplingLayer

class RealNVP(nn.Module):
    def __init__(self,n_c_layers = 8):
        super(RealNVP, self).__init__()
        assert n_c_layers > 1, "Need more than one coupling layer to transform both dimensions of x."
        
        self.base = torch.distributions.MultivariateNormal(torch.zeros(2), torch.eye(2))
        
        self.coupling_layers = []
        for layer in range(n_c_layers):
            self.coupling_layers.append(CouplingLayer(2,layer))
            
        self.model = nn.Sequential(*self.coupling_layers)

    def forward(self, x, reverse=False):
        if not reverse:
            sum_log_det_J = x.new_zeros(x.size(0))
                                   
            for i in range(len(self.coupling_layers)):
                coupling_layer = self.coupling_layers[i]
                x, log_det_J = coupling_layer(x,reverse)
                sum_log_det_J += log_det_J

            return x, sum_log_det_J
        else:
            for i in reversed(range(len(self.coupling_layers))):
                coupling_layer = self.coupling_layers[i]
                x = coupling_layer(x,reverse)
            return x