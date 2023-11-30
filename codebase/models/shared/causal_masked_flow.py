import torch
import torch.nn as nn
from torch.distributions import MultivariateNormal, Normal, Uniform
import torch.nn.functional as F
# device = torch.device("cuda:5" if(torch.cuda.is_available()) else "cpu")
from codebase import utils as ut


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
    
    
# FOR DIFFEOMORPHIC SCM-VAE
class MultivariateCausalFlow(nn.Module):
    def __init__(self, dim, k, C=None, net_class=MLP, nh=100, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.k = k

        self.C = C
        self.A = (torch.eye(self.C.shape[0]) - self.C)
        
        if scale:
            self.s_cond = net_class(self.dim*self.k, self.k, 100)
        if shift: 
            self.t_cond = net_class(self.dim*self.k, self.k, 100)
        
        self.z_int_prior = Normal(0.0, 1.0)

    
    def forward(self, e, target=None, value=None):

        total_dims = e.shape[1]*e.shape[2]
        log_det = torch.zeros(e.size(0)).to(e.device)
        p_logprob = torch.zeros(e.size(0)).to(e.device)
        batch_size = e.shape[0]
        z = torch.zeros(batch_size, self.dim, self.k).to(e.device)
        
        
        for i in range(self.dim):    
            if 1 in self.C[:, i]: # does it have any parents (z_3)
                # mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
                mask = self.C[:, i].repeat(self.k, 1).T.reshape(total_dims).to(e.device)
            elif 1 not in self.C[:, i] or target == i: # doesnt have parents
                mask = torch.zeros(total_dims).to(e.device)
            
            # compute slope and offset
            s = self.s_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.k) # slope
            t = self.t_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.k) # offset

            # slope and offset transformation (affine transformation)
            z[:, i, :] = torch.exp(s) * e[:, i, :].reshape(batch_size, self.k) + t
            if target is not None and value is not None:
                # temp = z.reshape(batch_size, self.dim*self.k)
                # temp[:, 77] = 0.1
                # z = temp.reshape(batch_size, self.dim, self.k)
                # temp = z.clone()
                # temp[:, 2, 19] = value[:, 19]
                # z = temp.clone()
                z[:, target, :] = value
                #z[:, 0, :] = value
            log_det += torch.sum(s, dim=1) # dz / de
            
        return z, log_det
    
    def backward(self, z, target=None, value=None):
        
        total_dims = z.shape[1]*z.shape[2]
        log_det = torch.zeros(z.size(0)).to(z.device)
        p_logprob = torch.zeros(z.size(0)).to(z.device)
        batch_size = z.shape[0]
        e = torch.zeros(batch_size, self.dim, self.k).to(z.device)
        
        
        for i in range(self.dim):
            
            if 1 in self.C[:, i]: # does it have any parents (z_3)
                # mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
                mask = self.C[:, i].repeat(self.k, 1).T.reshape(total_dims).to(e.device)
            elif 1 not in self.C[:, i] or target == i: # doesnt have parents
                mask = torch.zeros(total_dims).to(e.device)
            
            # compute slope and offset
            s = self.s_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.k) # slope
            t = self.t_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.k) # offset
            
        
            # slope and offset transformation (affine transformation)
            e[:, i, :] = torch.exp(-s) * (z[:, i, :].reshape(batch_size, self.k) - t)
            # if target is not None and value is not None:
            #     z[:, target, :] = torch.ones(1, self.k).to(e.device) * value
                # z[:, target, :] = value.to(device)
            log_det -= torch.mean(s, dim=1) # dz / de
            
        return z, log_det
    
    
    
    def forward_interv(self, e, I):
        total_dims = e.shape[1]*e.shape[2]
        log_det = torch.zeros(e.size(0)).to(e.device)
        p_logprob = torch.zeros(e.size(0)).to(e.device)
        batch_size = e.shape[0]
        z = torch.zeros(batch_size, self.dim, self.dim).to(e.device)
        
        
        for i in range(self.dim):
            
            interv_mask = (I[:, i] == 1).to(e.device) #[T, F, F, F]
            # print(interv_mask)
            
            if 1 in self.C[:, i]: # does it have any parents (z_3)
                # mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
                mask = self.C[:, i].repeat(4, 1).T.reshape(total_dims).to(e.device)
            else: # doesnt have parents
                mask = torch.zeros(total_dims).to(e.device)
            
            # Standard Gaussian sampled intervention
            #z_base = torch.randn(e[:, i].shape).to(device)
            
            # z_base = torch.randn(e[:, i, :].shape).to(device)
            
            # if z_base_inf is not None:
            #     z_base = z_base_inf

            # Intervention
            # e[:, i] = torch.where(interv_mask.reshape(batch_size), z_base.clone(), e[:, i].clone())
            # z[:, i, :] = torch.where(interv_mask.reshape(batch_size, 4), z_base.clone(), z[:, i, :].clone())
            z[:, i, :] = torch.ones(1, 4).to(e.device) * 3
  
            s = torch.where(interv_mask.reshape(batch_size, 4), 
                            self.s_cond(z.reshape(-1, total_dims), torch.zeros(total_dims).to(e.device)).reshape(batch_size, self.dim), 
                            self.s_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.dim))
            
            t = torch.where(interv_mask.reshape(batch_size, 4), 
                            self.t_cond(z.reshape(-1, total_dims), torch.zeros(total_dims).to(e.device)).reshape(batch_size, self.dim), 
                            self.t_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.dim))
        
            
            # slope and offset transformation (affine transformation)
            z[:, i, :] = torch.exp(s) * (e[:, i, :] - t)
            

        return z
    

    
# FOR DIFFEOMORPHIC SCM-VAE
class PriorMultivariateCausalFlow(nn.Module):
    def __init__(self, dim, k, C=None, net_class=MLP, nh=100, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        self.k = k

        self.C = C
        self.A = (torch.eye(self.C.shape[0]) - self.C)
        
        if scale:
            self.s_cond = net_class(self.dim*self.k, self.k, 100)
        if shift: 
            self.t_cond = net_class(self.dim*self.k, self.k, 100)
        
        self.z_int_prior = Normal(0.0, 1.0)

    
    def forward(self, e, latent=None, target=None, value=None):

        total_dims = e.shape[1]*e.shape[2]
        log_det = torch.zeros(e.size(0)).to(e.device)
        p_logprob = torch.zeros(e.size(0)).to(e.device)
        batch_size = e.shape[0]
        z = torch.zeros(batch_size, self.dim, self.k).to(e.device)
        
        
        for i in range(self.dim):    
            if 1 in self.C[:, i]: # does it have any parents (z_3)
                # mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
                mask = self.C[:, i].repeat(self.k, 1).T.reshape(total_dims).to(e.device)
            elif 1 not in self.C[:, i] or target == i: # doesnt have parents
                mask = torch.zeros(total_dims).to(e.device)
            
            # compute slope and offset
            s = self.s_cond(latent.reshape(-1, total_dims), mask).reshape(batch_size, self.k) # slope
            t = self.t_cond(latent.reshape(-1, total_dims), mask).reshape(batch_size, self.k) # offset

            # slope and offset transformation (affine transformation)
            z[:, i, :] = torch.exp(s) * e[:, i, :].reshape(batch_size, self.k) + t
            if target is not None and value is not None:
                # temp = z.reshape(batch_size, self.dim*self.k)
                # temp[:, 77] = 0.1
                # z = temp.reshape(batch_size, self.dim, self.k)
                # temp = z.clone()
                # temp[:, 2, 19] = value[:, 19]
                # z = temp.clone()
                z[:, target, :] = value
                #z[:, 0, :] = value
            log_det += torch.sum(s, dim=1) # dz / de
            
        return z, log_det
    
    def backward(self, z, target=None, value=None):
        
        total_dims = z.shape[1]*z.shape[2]
        log_det = torch.zeros(z.size(0)).to(z.device)
        p_logprob = torch.zeros(z.size(0)).to(z.device)
        batch_size = z.shape[0]
        e = torch.zeros(batch_size, self.dim, self.k).to(z.device)
        
        
        for i in range(self.dim):
            
            if 1 in self.C[:, i]: # does it have any parents (z_3)
                # mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
                mask = self.C[:, i].repeat(self.k, 1).T.reshape(total_dims).to(e.device)
            elif 1 not in self.C[:, i] or target == i: # doesnt have parents
                mask = torch.zeros(total_dims).to(e.device)
            
            # compute slope and offset
            s = self.s_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.k) # slope
            t = self.t_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.k) # offset
            
        
            # slope and offset transformation (affine transformation)
            e[:, i, :] = torch.exp(-s) * (z[:, i, :].reshape(batch_size, self.k) - t)
            # if target is not None and value is not None:
            #     z[:, target, :] = torch.ones(1, self.k).to(e.device) * value
                # z[:, target, :] = value.to(device)
            log_det -= torch.mean(s, dim=1) # dz / de
            
        return z, log_det
    
    
    
    def forward_interv(self, e, I):
        total_dims = e.shape[1]*e.shape[2]
        log_det = torch.zeros(e.size(0)).to(e.device)
        p_logprob = torch.zeros(e.size(0)).to(e.device)
        batch_size = e.shape[0]
        z = torch.zeros(batch_size, self.dim, self.dim).to(e.device)
        
        
        for i in range(self.dim):
            
            interv_mask = (I[:, i] == 1).to(e.device) #[T, F, F, F]
            # print(interv_mask)
            
            if 1 in self.C[:, i]: # does it have any parents (z_3)
                # mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
                mask = self.C[:, i].repeat(4, 1).T.reshape(total_dims).to(e.device)
            else: # doesnt have parents
                mask = torch.zeros(total_dims).to(e.device)
            
            # Standard Gaussian sampled intervention
            #z_base = torch.randn(e[:, i].shape).to(device)
            
            # z_base = torch.randn(e[:, i, :].shape).to(device)
            
            # if z_base_inf is not None:
            #     z_base = z_base_inf

            # Intervention
            # e[:, i] = torch.where(interv_mask.reshape(batch_size), z_base.clone(), e[:, i].clone())
            # z[:, i, :] = torch.where(interv_mask.reshape(batch_size, 4), z_base.clone(), z[:, i, :].clone())
            z[:, i, :] = torch.ones(1, 4).to(e.device) * 3
  
            s = torch.where(interv_mask.reshape(batch_size, 4), 
                            self.s_cond(z.reshape(-1, total_dims), torch.zeros(total_dims).to(e.device)).reshape(batch_size, self.dim), 
                            self.s_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.dim))
            
            t = torch.where(interv_mask.reshape(batch_size, 4), 
                            self.t_cond(z.reshape(-1, total_dims), torch.zeros(total_dims).to(e.device)).reshape(batch_size, self.dim), 
                            self.t_cond(z.reshape(-1, total_dims), mask).reshape(batch_size, self.dim))
        
            
            # slope and offset transformation (affine transformation)
            z[:, i, :] = torch.exp(s) * (e[:, i, :] - t)
            

        return z

    
    
# FOR ILCM
class CausalAffineAutoregFlow(nn.Module):
    def __init__(self, dim, C, net_class=MLP, nh=100, scale=True, shift=True):
        super().__init__()
        self.dim = dim
        # self.s_cond = lambda x: x.new_zeros(x.size(0), self.dim)
        # self.t_cond = lambda x: x.new_zeros(x.size(0), self.dim)
        self.C = C
        if scale:
            self.s_cond = net_class(self.dim, 1, 100)
        if shift: 
            self.t_cond = net_class(self.dim, 1, 100)
        
        self.z_int_prior = Normal(0.0, 1.0)
        # self.z_int_prior = Uniform(0.0, 1.0)


    
    def forward(self, e):
        log_det = torch.zeros(e.size(0)).to(device)
        p_logprob = torch.zeros(e.size(0)).to(device)
        batch_size = e.shape[0]
        z = torch.zeros(e.shape).to(device)
        
        # set z to e
        # z = e.clone()
        
        for i in range(self.dim):
            
            if 1 in self.C[:, i]: # does it have any parents (z_3)
                mask = self.C[:, i].reshape(self.dim).to(device) # [1, 1, 0, 0]
            else: # doesnt have parents
                mask = torch.zeros(self.dim).to(device)
            
            # compute slope and offset
            s = self.s_cond(z, mask).reshape(z.shape[0]) # slope
            t = self.t_cond(z, mask).reshape(z.shape[0]) # offset
            
            # print(s.shape)
            # print(z[:, i].shape)
            
            # slope and offset transformation (affine transformation)
            z[:, i] = torch.exp(s) * e[:, i] + t # z1 = s * e_1 + t, z_3 = s * e_3 + t
            # print(s)
            # f1(e_1, pai=0) = s*e_1 + t = z1
            # f2 --- z2
            # f3(e_3, pai = (z_1, z_2)) = s * e_3 + t, [z1, z2, 0, 0]
            # f4
            
            log_det += s # dz / de
            
        return z, log_det
    
    
    def backward(self, z, I, z_base_inf=None):
        log_det = torch.zeros(z.size(0)).to(device)
        p_logprob = torch.zeros(z.size(0)).to(device)
        batch_size = z.shape[0]
        e = torch.zeros(z.shape).to(device)
        
        # [e1, e2, e3, e4] = [z1, z2, z3, z4]
        # e = z.clone()
        
        # [z1, z2, e3, e4]
        # [z1, z2, z3, e4]
        # [z1, z2, z3, z4]
        # []
#         z_base = torch.randn(batch_size).to(device)
            
#         if z_base_inf is not None:
#             z_base = z_base_inf
#         z_base = torch.randn(batch_size).to(device)
            
#         if z_base_inf is not None:
#             z_base = z_base_inf
        for i in range(self.dim):
            
            interv_mask = (I[:, i] == 1).unsqueeze(-1).to(device) #[T, F, F, F]
            
            if 1 in self.C[:, i]: # if it has parents
                mask = self.C[:, i].reshape(self.dim).to(device)
            else: # if it doesnt
                mask = torch.zeros(self.dim).to(device)
            
            
            # Standard Gaussian sampled intervention
            #z_base = torch.randn(e[:, i].shape).to(device)
            
            z_base = torch.randn(e[:, i].shape).to(device)
            
            if z_base_inf is not None:
                z_base = z_base_inf

            # Intervention
            # e[:, i] = torch.where(interv_mask.reshape(batch_size), z_base.clone(), e[:, i].clone())
            z[:, i] = torch.where(interv_mask.reshape(batch_size), z_base.clone(), z[:, i].clone())

            # z3 = z3'
            
            # compute slope and offset as a function of e\i
#             s = torch.where(interv_mask.reshape(batch_size), 
#                             self.s_cond(e, torch.zeros(self.dim).to(device)).reshape(z.shape[0]), 
#                             self.s_cond(e, mask).reshape(z.shape[0]))
            
#             t = torch.where(interv_mask.reshape(batch_size), 
#                             self.t_cond(e, torch.zeros(self.dim).to(device)).reshape(z.shape[0]), 
#                             self.t_cond(e, mask).reshape(z.shape[0]))
            
            s = torch.where(interv_mask.reshape(batch_size), 
                            self.s_cond(z, torch.zeros(self.dim).to(device)).reshape(z.shape[0]), 
                            self.s_cond(z, mask).reshape(z.shape[0]))
            
            t = torch.where(interv_mask.reshape(batch_size), 
                            self.t_cond(z, torch.zeros(self.dim).to(device)).reshape(z.shape[0]), 
                            self.t_cond(z, mask).reshape(z.shape[0]))
        
            
            # slope and offset transformation (affine transformation)
            e[:, i] = torch.exp(-s) * (z[:, i] - t)
            
            s_new = torch.where(interv_mask.reshape(batch_size), s.to(device), torch.zeros(s.shape).to(device))
            z_val = torch.where(interv_mask.reshape(batch_size), self.z_int_prior.log_prob(z_base).to(device), torch.zeros(z[:, i].shape).to(device))
            
            # s = self.s_cond(z, mask).reshape(z.shape[0])
            # t = self.t_cond(z, mask).reshape(z.shape[0])
            
            log_det -= s_new
            p_logprob += z_val
            # p_logprob += ut.gaussian_log_prob(z_val, torch.zeros(batch_size).to(device), torch.ones(batch_size).to(device))
            # p_logprob += ut.log_normal(z_val, torch.zeros(batch_size).to(device), torch.ones(batch_size).to(device)) 
            
        return e, p_logprob, log_det
                

        

 
    
    
                

    
    
    