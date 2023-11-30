import sys
sys.path.append('../')
import torch
import numpy as np
from codebase import utils as ut
from torch import nn
from torch.nn import functional as F
# device = torch.device("cuda:3" if(torch.cuda.is_available()) else "cpu")
from codebase.models.shared.mask import Encoder, Decoder_DAG, DagLayer, MaskLayer, ConvEncoder, ConvDec
from codebase.models.shared import MultivariateCausalFlow
from models.prior import DisentanglementPrior



class ICM_VAE(nn.Module):
    def __init__(self, name='icm_vae_cdp', 
                 dataset="synthetic", 
                 z_dim=16, 
                 z1_dim=4, 
                 z2_dim=4, 
                 C=None,
                 scale=None,
                 inference = False, 
                 alpha=0, 
                 beta=0):
        super().__init__()
        self.name = name
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim
        self.channel = 4
        self.scale = scale 
        self.beta = beta
        self.alpha = alpha

        if dataset == "synthetic":
            self.enc = Encoder(self.z_dim, self.channel)
            self.dec = Decoder_DAG(self.z_dim,self.z1_dim, self.z2_dim)
        else:
            self.enc = ConvEncoder(self.z_dim)
            self.dec = ConvDec(self.z1_dim, self.z2_dim, self.z_dim)

        self.C = C
        
        # CAUSAL FLOW
        self.causal_flow = MultivariateCausalFlow(dim=self.z1_dim, k=self.z2_dim, C=self.C)

        # CAUSAL DISENTANGLEMENT PRIOR
        self.prior = DisentanglementPrior(dim=self.z1_dim, k=self.z2_dim, C=self.C)


    def forward(self, x, label, mask = None, traversal=None, value=None, sample = False, adj = None, alpha=0.1, beta=1, lambdav=0.001):
        """
        Computes the Evidence Lower Bound, KL and, Reconstruction costs

        Args:
            x: tensor: (batch, dim): Observations

        Returns:
            nelbo: tensor: (): Negative evidence lower bound
            kl: tensor: (): ELBO KL divergence to prior
            rec: tensor: (): ELBO Reconstruction term
        """
        #assert label.size()[1] == self.z1_dim

        # ENCODE TO REPRESENTATION
        eps_m, eps_v = self.enc.encode(x)

        eps_m = eps_m.reshape([eps_m.size()[0], self.z1_dim, self.z2_dim])  # RESHAPE TO (BATCH, 4, 4)

        eps_v = torch.ones([eps_m.size()[0], self.z1_dim, self.z2_dim]).to(x.device)

        # INTERVENTIONS - DURING INFERENCE
        if mask is not None:
            z_m, log_det_z = self.causal_flow(eps_m, target=mask, value=value)
            z_m[:, 3, :] = torch.abs(z_m[:, 3, :].clone())
            z_v = torch.zeros(z_m.shape).to(x.device)
        elif traversal is not None:
            z_m, log_det_z = self.causal_flow(eps_m)
            z_m[:, traversal, :] = value
            z_v = torch.zeros(z_m.shape).to(x.device)
        else:
            z_m, log_det_z = self.causal_flow(eps_m)
            z_v = torch.zeros(z_m.shape).to(x.device)
        
        # CAUSAL REPRESENTATION
        z_given_dag = ut.conditional_sample_gaussian(z_m, z_v * lambdav)

        x_hat, _, _, _, _ = self.dec.decode_sep(
            z_given_dag.reshape([z_given_dag.size()[0], self.z_dim]), label.to(x.device))

        # RECONSTRUCTION LOSS
        rec = ut.log_bernoulli_with_logits(x, x_hat.reshape(x.size()))
        rec = -torch.mean(rec)
        
        # PRIORS
        p_m, p_v = torch.zeros(z_m.size()), torch.ones(z_m.size())
        cp_m, cp_v = ut.condition_prior(self.scale, label, self.z2_dim)
        cp_m = self.prior(label.to(x.device), cp_m.to(x.device), z=z_m)

        cp_v = torch.ones([z_m.size()[0], self.z1_dim, self.z2_dim]).to(x.device)
        cp_z = ut.conditional_sample_gaussian(cp_m.to(x.device), cp_v.to(x.device))

        # KL-DIVERGENCE BETWEEN DISTRIBUTION FROM ENCODER AND THE ISOTROPIC GAUSSIAN PRIOR
        kl = torch.zeros(1).to(x.device)

        # RESHAPE
        eps_m = eps_m.view(-1, self.z_dim).to(x.device)
        eps_v = eps_v.view(-1, self.z_dim).to(x.device)
        p_m = p_m.view(-1, self.z_dim).to(x.device)
        p_v = p_v.view(-1, self.z_dim).to(x.device)

        kl = self.alpha * (ut.kl_normal(eps_m, eps_v, p_m, p_v) - log_det_z)
        
        # log_prior_prob = p_u + log_det_u
        for i in range(self.z1_dim):
            kl = kl + self.beta * ut.kl_normal(z_m[:, i, :].to(x.device), cp_v[:, i, :].to(x.device),
                                           cp_m[:, i, :].to(x.device), cp_v[:, i, :].to(x.device))

        kl = torch.mean(kl)

        neg_elbo = rec + kl
        # z_given_dag = ut.conditional_sample_gaussian(z_m, z_v * lambdav)

        return neg_elbo, kl, rec, x_hat.reshape(x.size()), z_given_dag, cp_m

    def loss(self, x):
        nelbo, kl, rec = self.negative_elbo_bound(x)
        loss = nelbo

        summaries = dict((
            ('train/loss', nelbo),
            ('gen/elbo', -nelbo),
            ('gen/kl_z', kl),
            ('gen/rec', rec),
        ))

        return loss, summaries

    def sample_sigmoid(self, batch):
        z = self.sample_z(batch)
        return self.compute_sigmoid_given(z)

    def compute_sigmoid_given(self, z):
        logits = self.dec.decode(z)
        return torch.sigmoid(logits)

    def sample_z(self, batch):
        return ut.sample_gaussian(
            self.z_prior[0].expand(batch, self.z_dim),
            self.z_prior[1].expand(batch, self.z_dim))

    def sample_x(self, batch):
        z = self.sample_z(batch)
        return self.sample_x_given(z)

    def sample_x_given(self, z):
        return torch.bernoulli(self.compute_sigmoid_given(z))
