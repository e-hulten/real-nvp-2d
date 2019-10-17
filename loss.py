import torch
import torch.nn as nn

class NegLogLik(nn.Module):
    def __init__(self, model):
        super(NegLogLik, self).__init__()
        self.base = model.base
        
    def __call__(self, z, sum_log_det_J):
        log_p = self.base.log_prob(z)
        return -(log_p + sum_log_det_J).mean()