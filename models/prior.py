import sys
sys.path.append('../')
import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Uniform
import torch.nn.functional as F
from codebase import utils as ut
from codebase.models.shared import PriorMultivariateCausalFlow, MultivariateCausalFlow
import numpy as np



class MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(nin, nh),
            nn.ReLU(),
            nn.Linear(nh, nh),
            nn.ReLU(),
            nn.Linear(nh, nout),
            nn.Sigmoid(),
        )
    def forward(self, x, mask):
        return self.net(x * mask)
    

class Norm_MLP(nn.Module):
    """ a simple 4-layer MLP """

    def __init__(self, nin, nout, nh):
        super().__init__()
        self.net = nn.Sequential(
            nn.BatchNorm1d(nin),
            nn.Linear(nin, nh),
            nn.GELU(),
            nn.Linear(nh, nh),
            nn.GELU(),
            nn.Linear(nh, nout),
            nn.Sigmoid(),
        )
    def forward(self, x, mask):
        return self.net(x * mask)
    

class DisentanglementPrior(nn.Module):
    def __init__(self, dim, k, C=None, nh=100, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.k = k
        self.C = C

        self.disentanglement_flow = PriorMultivariateCausalFlow(dim=self.dim, k=self.k, C=self.C, net_class=MLP)

        self.scale = np.array([[0,44],[100,40],[6.5, 3.5],[10,5]])
        
        self.z_int_prior = Normal(0.0, 1.0)

    
    def forward(self, u, u_ext, z=None):
        batch_size = u.shape[0]
        p_u = torch.zeros(batch_size).to(u.device)
        
        z_prior, log_det = self.disentanglement_flow(u_ext, z)
        
        return z_prior
    
                