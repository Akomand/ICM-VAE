import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear

import numpy as np
import torch
import torch.nn.functional as F
from codebase import utils as ut
from torch import autograd, nn, optim
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear

# Gaussian Encoder
class Encoder(nn.Module):
    def __init__(self, z_dim, channel=4, y_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.channel = channel
        self.fc1 = nn.Linear(self.channel * 96 * 96, 300)
        self.fc2 = nn.Linear(300 + y_dim, 300)
        self.fc3 = nn.Linear(300, 300)
        self.fc4 = nn.Linear(300, 2 * z_dim)
        self.LReLU = nn.LeakyReLU(0.2, inplace=True)
        self.net = nn.Sequential(
            nn.Linear(self.channel * 96 * 96, 900),
            nn.ELU(),
            nn.Linear(900, 300),
            nn.ELU(),
            nn.Linear(300, 2 * z_dim),
        )
        
        # self.net = nn.Sequential(
        #     nn.Linear(self.channel * 96 * 96, 900),
        #     nn.ELU(),
        #     nn.Linear(900, 300),
        # )
        
        self.fc_mu = nn.Linear(300, z_dim)
        self.fc_var = nn.Linear(300, z_dim)

    def conditional_encode(self, x, l):
        x = x.view(-1, self.channel * 96 * 96)
        x = F.elu(self.fc1(x))
        l = l.view(-1, 4)
        x = F.elu(self.fc2(torch.cat([x, l], dim=1)))
        x = F.elu(self.fc3(x))
        x = self.fc4(x)
        m, v = ut.gaussian_parameters(x, dim=1)
        return m, v

    def encode(self, x, y=None):
        xy = x if y is None else torch.cat((x, y), dim=1)
        xy = xy.view(-1, self.channel * 96 * 96)
        h = self.net(xy)
        
        # m = self.fc_mu(h)
        # v = self.fc_var(h)
        
        m, v = ut.gaussian_parameters(h, dim=1)
        
        return m, v

    
class Decoder(nn.Module):
    def __init__(self, z_dim, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.net = nn.Sequential(
            nn.Linear(z_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 4 * 96 * 96)
        )

    def decode(self, z, y=None):
        zy = z if y is None else torch.cat((z, y), dim=1)
        return self.net(zy)


    

    
    
class Decoder_DAG(nn.Module):
    def __init__(self, z_dim, concept, z1_dim, channel=4, y_dim=0):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept
        self.y_dim = y_dim
        self.channel = channel
        # print(self.channel)
        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim + y_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )
        self.net5 = nn.Sequential(
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )

        self.net6 = nn.Sequential(
            nn.Linear(z_dim, 300),
            nn.ELU(),
            nn.Linear(300, 300),
            nn.ELU(),
            nn.Linear(300, 1024),
            nn.ELU(),
            nn.Linear(1024, 1024),
            nn.ELU(),
            nn.Linear(1024, self.channel * 96 * 96)
        )

    def decode_condition(self, z, u):
        # z = z.view(-1,3*4)
        z = z.view(-1, 4 * 4)
        z1, z2, z3, z4 = torch.split(z, self.z_dim // 4, dim=1)
        # print(z1.shape)
        # exit(0)
        # print(u[:,0].reshape(1,u.size()[0]).size())
        rx1 = self.net1(
            torch.transpose(torch.cat((torch.transpose(z1, 1, 0), u[:, 0].reshape(1, u.size()[0])), dim=0), 1, 0))
        rx2 = self.net2(
            torch.transpose(torch.cat((torch.transpose(z2, 1, 0), u[:, 1].reshape(1, u.size()[0])), dim=0), 1, 0))
        rx3 = self.net3(
            torch.transpose(torch.cat((torch.transpose(z3, 1, 0), u[:, 2].reshape(1, u.size()[0])), dim=0), 1, 0))
        rx4 = self.net4(
            torch.transpose(torch.cat((torch.transpose(z4, 1, 0), u[:, 2].reshape(1, u.size()[0])), dim=0), 1, 0))
        temp = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        # print(temp.shape)
        # exit(0)
        # h = self.net6(torch.cat((rx1, rx2, rx3, rx4), dim=1))

        h = (rx1 + rx2 + rx3 + rx4) / 4

        return h

    def decode_mix(self, z):
        z = z.permute(0, 2, 1)
        z = torch.sum(z, dim=2, out=None)
        # print(z.contiguous().size())
        z = z.contiguous()
        h = self.net1(z)
        return h

    def decode_union(self, z, u, y=None):

        z = z.view(-1, self.concept * self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
        else:
            zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = self.net5((rx1 + rx2 + rx3 + rx4) / 4)
        return h, h, h, h, h

    def decode(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        h = self.net6(z)
        return h, h, h, h, h

    def decode_sep(self, z, u, y=None):
        z = z.view(-1, self.concept * self.z1_dim)
        zy = z if y is None else torch.cat((z, y), dim=1)

        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
            elif self.concept == 3:
                zy1, zy2, zy3 = zy[:, 0], zy[:, 1], zy[:, 2]
        else:
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
            elif self.concept == 3:
                zy1, zy2, zy3 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept == 4:
            rx4 = self.net4(zy4)
            h = (rx1 + rx2 + rx3 + rx4) / self.concept
        elif self.concept == 3:
            h = (rx1 + rx2 + rx3) / self.concept

        return h, h, h, h, h

    def decode_cat(self, z, u, y=None):
        z = z.view(-1, 4 * 4)
        zy = z if y is None else torch.cat((z, y), dim=1)
        zy1, zy2, zy3, zy4 = torch.split(zy, 1, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = self.net5(torch.cat((rx1, rx2, rx3, rx4), dim=1))
        return h
    
    
    
class F_SCM(nn.Module):
    def __init__(self, latent_dim=4, f_dim=4):
        super().__init__()
        self.latent_dim = latent_dim
        self.f_dim = f_dim
        self.net = nn.Sequential(
            nn.Linear(self.latent_dim*self.f_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 32),
            nn.ReLU(),
            nn.Linear(32, f_dim)
        )

    def forward(self, z, z_int, mask, I=None):
        # print(z.shape)
        # exit(0)
        z_masked = torch.empty(z.size()).to(device)
        for i in range(4):
            # if 1 in mask[:, i]:
            #     eps = torch.normal(mean=torch.zeros(self.f_dim), std=torch.ones(self.f_dim)).to(device)
            #     z_masked[:, i] = self.net((z * mask[:, i]).reshape(-1, self.latent_dim*self.f_dim)) + eps
            # else:
            #     z_masked[:, i] = z[:, i]

            if I is not None:
                for j in range(z.shape[0]):
                    if I[j][0][i] == 0:
                        z_masked[j, i] = z[j, i]
                    else:
                        z_masked[j, i] = z_int[j, i]

            if 1 in mask[:, i]:
                eps = torch.normal(mean=torch.zeros(self.f_dim), std=torch.ones(self.f_dim)).to(device)
                z_masked[:, i] = self.net((z * mask[:, i]).reshape(-1, self.latent_dim*self.f_dim)) + eps
            else:
                z_masked[:, i] = z[:, i]
        # exit(0)
        # mean, std, var = torch.mean(z_masked), torch.std(z_masked), torch.var(z_masked)

        return z_masked


class MaskLayer(nn.Module):
    def __init__(self, z_dim, concept=4, z1_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ELU(),
            nn.Linear(32, z_dim),
        )
        
    def masked(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    def masked_sep(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z


    def mix(self, z):
        zy = z.view(-1, self.concept * self.z1_dim)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
            elif self.concept == 3:
                zy1, zy2, zy3 = zy[:, 0], zy[:, 1], zy[:, 2]
        else:
            if self.concept == 4:
                #print(zy.shape)
                #print(len(torch.split(zy, self.z_dim // self.concept, dim=1)))
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
            elif self.concept == 3:
                zy1, zy2, zy3 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept == 4:
            rx4 = self.net4(zy4)
            h = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        elif self.concept == 3:
            h = torch.cat((rx1, rx2, rx3), dim=1)
        # print(h.size())
        return h

    
    
class MaskLayer1(nn.Module):
    def __init__(self, z_dim, concept=4, z1_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ELU(),
            nn.Linear(32, z_dim),
        )
        self.net_g = nn.Sequential(
            nn.Linear(z_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )

    def masked(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    def masked_sep(self, z):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z

    def g(self, z, i=None):
        # z = z[:, :8]
        # print(z.shape)
        # exit(0)
        rx = self.net_g(z)

        # print(rx.shape)
        # exit(0)
        return rx

    def mix(self, z):
        zy = z.view(-1, self.concept * self.z1_dim)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            if self.concept == 4:
                zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
            elif self.concept == 3:
                zy1, zy2, zy3 = zy[:, 0], zy[:, 1], zy[:, 2]
        else:
            if self.concept == 4:
                #print(zy.shape)
                #print(len(torch.split(zy, self.z_dim // self.concept, dim=1)))
                zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
            elif self.concept == 3:
                zy1, zy2, zy3 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        if self.concept == 4:
            rx4 = self.net4(zy4)
            h = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        elif self.concept == 3:
            h = torch.cat((rx1, rx2, rx3), dim=1)
        # print(h.size())
        return h
    
    

class Mix(nn.Module):
    def __init__(self, z_dim, concept, z1_dim):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim, 16),
            nn.ELU(),
            nn.Linear(16, z1_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim, 16),
            nn.ELU(),
            nn.Linear(16, z1_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim, 16),
            nn.ELU(),
            nn.Linear(16, z1_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim, 16),
            nn.ELU(),
            nn.Linear(16, z1_dim),
        )

    def mix(self, z):
        zy = z.view(-1, self.concept * self.z1_dim)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
        else:
            zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        # print(h.size())
        return h


class CausalLayer(nn.Module):
    def __init__(self, z_dim, concept=4, z1_dim=4):
        super().__init__()
        self.z_dim = z_dim
        self.z1_dim = z1_dim
        self.concept = concept

        self.elu = nn.ELU()
        self.net1 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net2 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net3 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim),
        )
        self.net4 = nn.Sequential(
            nn.Linear(z1_dim, 32),
            nn.ELU(),
            nn.Linear(32, z1_dim)
        )
        self.net = nn.Sequential(
            nn.Linear(z_dim, 128),
            nn.ELU(),
            nn.Linear(128, z_dim),
        )

    def calculate(self, z, v):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z, v

    def masked_sep(self, z, v):
        z = z.view(-1, self.z_dim)
        z = self.net(z)
        return z, v

    def calculate_dag(self, z, v):
        zy = z.view(-1, self.concept * self.z1_dim)
        if self.z1_dim == 1:
            zy = zy.reshape(zy.size()[0], zy.size()[1], 1)
            zy1, zy2, zy3, zy4 = zy[:, 0], zy[:, 1], zy[:, 2], zy[:, 3]
        else:
            zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1(zy1)
        rx2 = self.net2(zy2)
        rx3 = self.net3(zy3)
        rx4 = self.net4(zy4)
        h = torch.cat((rx1, rx2, rx3, rx4), dim=1)
        # print(h.size())
        return h, v


class Attention(nn.Module):
    def __init__(self, in_features, bias=False):
        super().__init__()
        self.M = nn.Parameter(torch.nn.init.normal_(torch.zeros(in_features, in_features), mean=0, std=1))
        self.sigmd = torch.nn.Sigmoid()

    # self.M =  nn.Parameter(torch.zeros(in_features,in_features))
    # self.A = torch.zeros(in_features,in_features).to(device)

    def attention(self, z, e):
        a = z.matmul(self.M).matmul(e.permute(0, 2, 1))
        a = self.sigmd(a)
        # print(self.M)
        A = torch.softmax(a, dim=1)
        e = torch.matmul(A, e)
        return e, A


class DagLayer(nn.Linear):
    def __init__(self, in_features, out_features, A=None, i=False, bias=False):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.i = i
        self.a = torch.zeros(out_features, out_features)
        self.a = self.a
        # self.a[0][1], self.a[0][2], self.a[0][3] = 1, 1, 1
        # self.a[1][2], self.a[1][3] = 1, 1
        # self.a[0, 2], self.a[1, 2], self.a[1, 3], self.a[3, 2] = 1, 1, 1, 1

        # self.a[0, 2:4] = 1
        # self.a[1, 2:4] = 1
        self.a = A

        # self.a[0, 1], self.a[1, 2], self.a[3, 2] = 1, 1, 1
        # self.a[0, 2], self.a[1, 3], self.a[2, 3] = 1, 1, 1

        # self.A = nn.Parameter(self.a)
        self.A = self.a#.to(device)

        self.b = torch.eye(out_features)
        self.b = self.b
        self.B = self.b#.to(device)
        # self.B = nn.Parameter(self.b)

        self.I = torch.eye((out_features))#.to(device)
        # self.I = nn.Parameter(torch.eye(out_features))
        # self.I.requires_grad = False
        if bias:
            self.bias = Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def mask_z(self, x, i):
        self.B = self.A.to(x.device)

        # x = torch.mul(self.B[:, i].reshape(4, 1).clone(), x.clone())
        x = torch.mul((self.B + self.I.to(x.device))[:, i].reshape(4, 1).clone(), x.clone())

        # x = torch.matmul(self.B.t().clone(), x.clone())
        # print(x.shape)
        # x = torch.matmul(self.B.t(), x)

        return x
    
    def mask_z_orig(self,x):
        self.B = self.A.to(x.device)
        #if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = torch.matmul(self.B.t().float(), x)
        return x

    def mask_z_learn(self, x, i):
        self.B = self.A.to(x.device)

        # x = F.linear(x.clone(), (self.B + self.I)[:, i].reshape(4, 1).clone(), self.bias)
        x = torch.mul((self.B + self.I.to(x.device))[:, i].reshape(4, 1).clone(), x.clone())

        return x


    def mask_u(self, x):
        self.B = self.A.to(x.device)
        # if self.i:
        #    x = x.view(-1, x.size()[1], 1)
        #    x = torch.matmul((self.B+0.5).t().int().float(), x)
        #    return x
        x = x.view(-1, x.size()[1], 1)
        x = torch.matmul(self.B.t(), x)
        return x

    def inv_cal(self, x, v):
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
        x = F.linear(x, self.I - self.A, self.bias)

        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        return x, v

    def calculate_dag(self, x, v):
        # print(self.A)
        # x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        self.A = self.A.to(x.device)
        if x.dim() > 2:
            x = x.permute(0, 2, 1)
        x = F.linear(x, torch.inverse(self.I - self.A.t()), self.bias)
        # print(x.size())

        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
        return x, v

    def calculate_cov(self, x, v):
        # print(self.A)
        v = ut.vector_expand(v)
        # x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        # print(v)
        return x, v

    def calculate_gaussian_ini(self, x, v):
        print(self.A)
        # x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)

        if x.dim() > 2:
            x = x.permute(0, 2, 1)
            v = v.permute(0, 2, 1)
        x = F.linear(x, torch.inverse(self.I - self.A), self.bias)
        v = F.linear(v, torch.mul(torch.inverse(self.I - self.A), torch.inverse(self.I - self.A)), self.bias)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
            v = v.permute(0, 2, 1).contiguous()
        return x, v

    # def encode_
    def forward(self, x):
        # x = x * torch.inverse((self.A) + self.I)

        if x.dim() > 2:
            x = x.permute(0, 2, 1)

        x = torch.matmul(x, torch.inverse(self.I.to(x.device) - self.A.t().to(x.device)).t())

        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()

        return x

    def calculate_gaussian(self, x, v):
        print(self.A)
        # x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)

        if x.dim() > 2:
            x = x.permute(0, 2, 1)
            v = v.permute(0, 2, 1)
        x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
        v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
        v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
        if x.dim() > 2:
            x = x.permute(0, 2, 1).contiguous()
            v = v.permute(0, 2, 1).contiguous()
        return x, v

    

class DagLayerOrig(nn.Linear):
	def __init__(self, in_features, out_features,i = False, bias=False):
		super(Linear, self).__init__()
		self.in_features = in_features
		self.out_features = out_features
		self.i = i
		# self.a = 0.5*torch.ones(out_features,out_features)
		self.a = torch.zeros(out_features,out_features)
		self.a = self.a
		#self.a[0][1], self.a[0][2], self.a[0][3] = 1,1,1
		#self.a[1][2], self.a[1][3] = 1,1

		self.a[0, 2:4], self.a[1, 2:4] = 1, 1

		# self.a[0, 2], self.a[1, 2], self.a[1, 3], self.a[3, 2] = 1, 1, 1, 1

		# self.a[0, 1], self.a[1, 2], self.a[3, 2] = 1, 1, 1
		# self.a[0, 2], self.a[2, 3], self.a[1, 3] = 1, 1, 1

		# self.A = nn.Parameter(self.a)
		self.A = self.a#.to(device)

		self.b = torch.eye(out_features)
		self.b = self.b
		# self.B = nn.Parameter(self.b)
		self.B = self.b#.to(device)

		self.I = torch.eye(out_features)#.to(device)
		# self.I = nn.Parameter(torch.eye(out_features))
		# self.I.requires_grad=False
		if bias:
			self.bias = Parameter(torch.Tensor(out_features))
		else:
			self.register_parameter('bias', None)

	def mask_z(self,x):
		self.B = self.A.to(x.device)
		#if self.i:
		#    x = x.view(-1, x.size()[1], 1)
		#    x = torch.matmul((self.B+0.5).t().int().float(), x)
		#    return x
		x = torch.matmul(self.B.t(), x)
		return x

	def mask_u(self,x):
		self.B = self.A.to(x.device)
		#if self.i:
		#    x = x.view(-1, x.size()[1], 1)
		#    x = torch.matmul((self.B+0.5).t().int().float(), x)
		#    return x
		x = x.view(-1, x.size()[1], 1)
		x = torch.matmul(self.B.t(), x)
		return x

	def inv_cal(self, x,v):
		if x.dim()>2:
			x = x.permute(0,2,1)
		x = F.linear(x, self.I - self.A, self.bias)

		if x.dim()>2:
			x = x.permute(0,2,1).contiguous()
		return x,v

	def calculate_dag(self, x):
		#print(self.A)
		#x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)

		if x.dim()>2:
			x = x.permute(0,2,1)
		x = F.linear(x, torch.inverse(self.I.to(x.device) - self.A.t().to(x.device)), self.bias)
		#print(x.size())

		if x.dim()>2:
			x = x.permute(0,2,1).contiguous()
		return x

	def calculate_cov(self, x, v):
		#print(self.A)
		v = ut.vector_expand(v)
		#x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)
		x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
		v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
		v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
		#print(v)
		return x, v

	def calculate_gaussian_ini(self, x, v):
		print(self.A)
		#x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)

		if x.dim()>2:
			x = x.permute(0,2,1)
			v = v.permute(0,2,1)
		x = F.linear(x, torch.inverse(self.I - self.A), self.bias)
		v = F.linear(v, torch.mul(torch.inverse(self.I - self.A),torch.inverse(self.I - self.A)), self.bias)
		if x.dim()>2:
			x = x.permute(0,2,1).contiguous()
			v = v.permute(0,2,1).contiguous()
		return x, v
	#def encode_
	def forward(self, x):
		# x = x * torch.inverse((self.A)+self.I)

		if x.dim() > 2:
			x = x.permute(0, 2, 1)

		x = torch.matmul(x, torch.inverse(self.I.to(x.device) - self.A.t().to(x.device)).t())

		if x.dim() > 2:
			x = x.permute(0, 2, 1).contiguous()

		return x
	def calculate_gaussian(self, x, v):
		print(self.A)
		#x = F.linear(x, torch.inverse((torch.abs(self.A))+self.I), self.bias)

		if x.dim()>2:
			x = x.permute(0,2,1)
			v = v.permute(0,2,1)
		x = dag_left_linear(x, torch.inverse(self.I - self.A), self.bias)
		v = dag_left_linear(v, torch.inverse(self.I - self.A), self.bias)
		v = dag_right_linear(v, torch.inverse(self.I - self.A), self.bias)
		if x.dim()>2:
			x = x.permute(0,2,1).contiguous()
			v = v.permute(0,2,1).contiguous()
		return x, v
    
    
    
    

    
# [(in - k + 2p)/s] + 1
class ConvEncoder(nn.Module):
    def __init__(self, out_dim=None):
        super().__init__()
        # init 128*128
        # 64x64x32, 32x32x64, 16x16x64, 8x8x64, 4x4x256, 4x4x3 (want
        # init 96*96
        self.conv1 = torch.nn.Conv2d(3, 32, 4, 2, 1)  # 48*48
        self.conv2 = torch.nn.Conv2d(32, 64, 4, 2, 1, bias=False)  # 24*24
        self.conv3 = torch.nn.Conv2d(64, 1, 4, 2, 1, bias=False)
        # self.conv4 = torch.nn.Conv2d(128, 1, 1, 1, 0) # 54*44

        self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.convm = torch.nn.Conv2d(1, 1, 4, 2, 1)
        self.convv = torch.nn.Conv2d(1, 1, 4, 2, 1)
        self.mean_layer = nn.Sequential(
            torch.nn.Linear(8 * 8, out_dim)
        )  # 12*12
        self.var_layer = nn.Sequential(
            torch.nn.Linear(8 * 8, out_dim)
        )
        # self.fc1 = torch.nn.Linear(6*6*128, 512)
        self.conv6 = nn.Sequential(
            nn.Conv2d(3, 32, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 2, 1), # 4x4x256
            nn.ReLU(True),
            nn.Conv2d(256, 64, 4, 2, 1) # 2x2x64
        )

    def encode(self, x):
        x = self.LReLU(self.conv1(x))
        x = self.LReLU(self.conv2(x))
        x = self.LReLU(self.conv3(x))
        # x = self.LReLU(self.conv4(x))
        # print(x.size())
        hm = self.convm(x)
        # print(hm.size())
        hm = hm.view(-1, 8 * 8)
        hv = self.convv(x)
        hv = hv.view(-1, 8 * 8)
        mu, var = self.mean_layer(hm), self.var_layer(hv)
        var = F.softplus(var) + 1e-8
        # var = torch.reshape(var, [-1, 16, 16])
        # print(mu.size())
        return mu, var

    def encode_simple(self, x):
        x = self.conv6(x)
        x = x.reshape(x.shape[0], 256)
        # print(x.shape)
        # exit(0)
        m, v = ut.gaussian_parameters(x, dim=1)
        # print(m.size())
        return m, v

# [(in - k + 2p)/s] + 1 = out
class ConvDecoder(nn.Module):
    def __init__(self, z2_dim):
        super().__init__()
        self.z2_dim = z2_dim
        
        self.net6 = nn.Sequential(
            nn.Conv2d(self.z2_dim, 128, 1), # 1-1+0 / 1 = 1x1x128
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4), # (128 - 1)
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

    def decode_sep(self, x):
        return None

    def decode(self, z):
        z = z.view(-1, self.z2_dim, 1, 1)
        z = self.net6(z)
        return z


class ConvDec(nn.Module):
    def __init__(self, concept, z1_dim, z_dim):
        super().__init__()
        self.concept = concept
        self.z1_dim = z1_dim
        self.z_dim = z_dim
        self.net1 = ConvDecoder(z1_dim)
        self.net2 = ConvDecoder(z1_dim)
        self.net3 = ConvDecoder(z1_dim)
        self.net4 = ConvDecoder(z1_dim)
        self.net5 = nn.Sequential(
            nn.Linear(16, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024)
        )
        self.net6 = nn.Sequential(
            nn.Conv2d(16, 128, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

    def decode_sep(self, z, u=None, y=None):
        z = z.view(-1, self.concept * self.z1_dim)

        zy = z if y is None else torch.cat((z, y), dim=1)
        zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)
        rx1 = self.net1.decode(zy1)
        # print(rx1.size())
        rx2 = self.net2.decode(zy2)
        rx3 = self.net3.decode(zy3)
        rx4 = self.net4.decode(zy4)
        z = (rx1 + rx2 + rx3 + rx4) / 4
        return z, z, z, z, z

    def decode(self, z, u=None, y=None):
        z = z.view(-1, self.concept * self.z1_dim, 1, 1)
        z = self.net6(z)
        # print(z.size())

        return z










#############################################################################################################
############################################# CELEBA ENC/DEC ################################################
#############################################################################################################

class CelebAConvEncoder(nn.Module):
    def __init__(self, latent_dim, in_channels=3, out_dim=None):
        super().__init__()
        self.latent_dim = latent_dim
        # 128x128
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, 4, 2, 1),            # 64x64x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(32, 64, 4, 2, 1),                     # 32x32x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 64, 4, 2, 1),                     # 16x16x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),                    # 8x8x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 128, 4, 2, 1),                   # 4x4x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 1),                      # 1x1x256
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 1)                    # 1x1x512
        )




        modules = []

        hidden_dims = [32, 64, 128, 256, 512, 512, 512]

        # in: 128x128x3, out: 64x64x32
        # in: 64x64x32, out: 32x32x64
        # out: 16x16x128
        # out: 8x8x256
        # out: 4x4x512

        # Build Encoder
        for h_dim in hidden_dims:
            modules.append(
                nn.Sequential(
                    nn.Conv2d(in_channels, out_channels=h_dim,
                              kernel_size=4, stride=2, padding=1),
                    nn.BatchNorm2d(h_dim),
                    nn.LeakyReLU(0.2, inplace=True))
            )
            in_channels = h_dim

        self.encoder = nn.Sequential(*modules)
        self.fc_mu = nn.Linear(hidden_dims[-1], latent_dim)
        self.fc_var = nn.Linear(hidden_dims[-1], latent_dim)


    def encode(self, x):
        z = self.conv(x)
        z = z.view(-1, 512)

        # Split the result into mu and var components
        # of the latent Gaussian distribution
        mu = self.fc_mu(z)
        var = self.fc_var(z)
        var = F.softplus(var) + 1e-8

        return mu, var


class CelebAConvDecoder(nn.Module):
    def __init__(self, latent_dim, out_channels=3, out_dim=None):
        super().__init__()
        self.latent_dim = latent_dim


        self.convT = nn.Sequential(
            nn.Conv2d(latent_dim, 512, 1),                      # 1x1x512
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(512, 256, 4),                    # 4x4x256
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),              # 8x8x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),              # 16x16x128
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),               # 32x32x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),                # 64x64x64
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),                # 128x128x32
            nn.LeakyReLU(0.2, inplace=True),
            nn.ConvTranspose2d(32, out_channels, 1),         # 128x128x3
        )


        modules = []

        hidden_dims = [32, 64, 128, 256, 512, 512, 512]

        # in: 128x128x3, out: 64x64x32
        # in: 64x64x32, out: 32x32x64
        # out: 16x16x128
        # out: 8x8x256
        # out: 4x4x512

        self.decoder_input = nn.Linear(latent_dim, hidden_dims[-1])

        hidden_dims.reverse()

        # in: 4x4x512, out: 8x8x256
        # in: 8x8x256, out: 16x16x128
        for i in range(len(hidden_dims) - 1):
            modules.append(
                nn.Sequential(
                    nn.ConvTranspose2d(hidden_dims[i],
                                       hidden_dims[i + 1],
                                       kernel_size=4,
                                       stride=2,
                                       padding=1),
                    nn.BatchNorm2d(hidden_dims[i + 1]),
                    nn.LeakyReLU(0.2, inplace=True))
            )

        self.decoder = nn.Sequential(*modules)

        self.final_layer = nn.Sequential(
            nn.ConvTranspose2d(hidden_dims[-1],
                               hidden_dims[-1],
                               kernel_size=4,
                               stride=2,
                               padding=1, output_padding=1),
            nn.BatchNorm2d(hidden_dims[-1]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_dims[-1], out_channels=out_channels,
                      kernel_size=4, padding=1),
            nn.Tanh())

    def decode(self, z):
        z = z.view(-1, self.latent_dim, 1, 1)
        z = self.convT(z)
        return z

    # def decode(self, z):
    #     z = self.decoder_input(z)
    #     z = z.view(-1, 512, 1, 1)
    #     x = self.decoder(z)
    #     x = self.final_layer(x)
    #
    #     return x


class CelebAConvDec(nn.Module):
    def __init__(self, latent_dim, out_dim=None):
        super().__init__()
        self.concept = 4
        self.z_dim = latent_dim
        self.z1_dim = self.z_dim // self.concept

        self.net1 = CelebAConvDecoder(self.z1_dim)
        self.net2 = CelebAConvDecoder(self.z1_dim)
        self.net3 = CelebAConvDecoder(self.z1_dim)
        self.net4 = CelebAConvDecoder(self.z1_dim)

    def decode_sep(self, z, u=None, y=None):
        z = z.view(-1, self.concept * self.z1_dim)  # 16x64

        zy = z if y is None else torch.cat((z, y), dim=1)
        # print(zy.shape)
        zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)  # each is 16x16
        rx1 = self.net1.decode(zy1)
        # print(f"Hi: {rx1.size()}")
        rx2 = self.net2.decode(zy2)
        rx3 = self.net3.decode(zy3)
        rx4 = self.net4.decode(zy4)
        # z = torch.cat((rx1, rx2, rx3, rx4), dim=0)
        z = (rx1+rx2+rx3+rx4)/4
        # print(z.shape)
        # sys.exit(0)
        return z, z, z, z, z






class ConvEncoderPend(nn.Module):
    def __init__(self, latent_dim, in_channel=3, out_dim=None):
        super().__init__()
        # init 96*96
        self.conv1 = torch.nn.Conv2d(in_channel, 24, 4, 2, 1)  # 48*48
        self.conv2 = torch.nn.Conv2d(24, 48, 4, 2, 1, bias=False)  # 24*24
        self.conv3 = torch.nn.Conv2d(48, 1, 4, 2, 1, bias=False)  # 12x12
        self.conv4 = torch.nn.Conv2d(1, 1, 3, 1, bias=False)
        # self.conv4 = torch.nn.Conv2d(128, 1, 1, 1, 0) # 54*44

        self.LReLU = torch.nn.LeakyReLU(0.2, inplace=True)
        self.convm = torch.nn.Conv2d(1, 1, 3, 1)  # 6x6 - BUT, changed padding from 1 to 3 in order to make this 8x8
        self.convv = torch.nn.Conv2d(1, 1, 3, 1)  # 6x6
        self.mean_layer = nn.Sequential(
            torch.nn.Linear(8*8, latent_dim)
        )  # 12*12
        self.var_layer = nn.Sequential(
            torch.nn.Linear(8*8, latent_dim)
        )

        # self.conv = nn.Sequential(
        #     nn.Conv2d(3, 32, 4, 2, 1),  # 48x48
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 32, 4, 2, 1),  # 24x24
        #     nn.ReLU(True),
        #     nn.Conv2d(32, 64, 4, 2, 1),  # 12x12
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),  # 6x6
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 64, 4, 2, 1),  # 3x3
        #     nn.ReLU(True),
        #     nn.Conv2d(64, 256, 4, 1),  # 2x2
        #     nn.ReLU(True),
        #     nn.Conv2d(256, 128, 1)  # 2x2
        # )


        self.conv = nn.Sequential(
            nn.Conv2d(in_channel, 24, 4, 2, 1),  # 48x48x24
            nn.LeakyReLU(0.2),
            nn.Conv2d(24, 24, 4, 2, 1),  # 24x24x24
            nn.LeakyReLU(0.2),
            nn.Conv2d(24, 48, 4, 2, 1),  # 12x12x48
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 48, 4, 2, 1),  # 6x6x48
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 48, 4, 2, 1),  # 3x3x48
            nn.LeakyReLU(0.2),
            nn.Conv2d(48, 96, 3, 1),  # 3x3x48
            nn.LeakyReLU(0.2),
            nn.Conv2d(96, latent_dim*2, 4, 2, 2) # 1x1x32
        )

    def encode(self, x):
        # print(x.shape)
        # sys.exit(0)
        x = self.LReLU(self.conv1(x))
        x = self.LReLU(self.conv2(x))
        x = self.LReLU(self.conv3(x))
        x = self.LReLU(self.conv4(x))

        # x = self.LReLU(self.conv4(x))
        # print(x.size())
        hm = self.convm(x)
        # print(hm.size())
        hm = hm.view(-1, 8 * 8)
        # print(hm.size())
        hv = self.convv(x)
        hv = hv.view(-1, 8 * 8)

        # print(hm.shape)
        # sys.exit(0)
        mu, var = self.mean_layer(hm), self.var_layer(hv)
        var = F.softplus(var) + 1e-8
        # var = torch.reshape(var, [-1, 16, 16])
        # print(mu.size())
        return mu, var

    def encode_simple(self, x):
        x = self.conv(x)
        # print(x.shape)
        # sys.exit(0)
        # x = x.view(-1, 96)
        # x = self.fc(x)
        # print(x.shape)
        m, v = ut.gaussian_parameters(x, dim=1)

        return m, v




# CONVOLUTIONAL DECODER LAYER
class ConvDecoderPend(nn.Module):
    def __init__(self, latent_dim, channels=3, out_dim=None):
        super().__init__()

        self.net6 = nn.Sequential(
            nn.Conv2d(latent_dim, 96, 1),  # 1x1x96
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(96, 48, 4),  # 3x3x48
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(48, 48, 2, 2, 1),  # 6x6
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(48, 24, 4, 2, 1),  # 12x12
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(24, 24, 4, 2, 1),  # 24x24x24
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(24, 24, 4, 2, 1),  # 48x48x12
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(24, channels, 4, 2, 1),  # 96x96x4
        )

        # self.net6 = nn.Sequential(
        #     nn.Conv2d(latent_dim, 96, 1), # 1x1x96
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(96, 48, 3, 1), # 3x3x48
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(48, 48, 3, 2, 1), # 6x6
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(48, 24, 3, 2, 1), # 12x12
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(24, 24, 3, 2, 1), # 24x24x24
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(24, 24, 3, 2, 1), # 48x48x12
        #     nn.LeakyReLU(0.2),
        #     nn.ConvTranspose2d(24, channels, 3, 2, 1), # 96x96x4
        # )

    def decode_sep(self, x):
        return None

    def decode(self, z):
        z = z.view(-1, z.shape[1], 1, 1)
        z = self.net6(z)
        return z



class ConvDecPend(nn.Module):
    def __init__(self, latent_dim, channels=3, out_dim=None):
        super().__init__()
        self.concept = 4
        self.z_dim = latent_dim
        self.z1_dim = self.z_dim // self.concept

        self.net1 = ConvDecoderPend(self.z1_dim, channels)
        self.net2 = ConvDecoderPend(self.z1_dim, channels)
        self.net3 = ConvDecoderPend(self.z1_dim, channels)
        self.net4 = ConvDecoderPend(self.z1_dim, channels)
        self.net5 = nn.Sequential(
            nn.Linear(16, 512),
            nn.BatchNorm1d(512),
            nn.Linear(512, 1024),
            nn.BatchNorm1d(1024)
        )
        self.net6 = nn.Sequential(
            nn.Conv2d(16, 128, 1),  # 4x4
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4),  # 1x1
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 32, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, 3, 4, 2, 1)
        )

    def decode_sep(self, z, u=None, y=None):
        z = z.view(-1, self.concept * self.z1_dim)  # 16x64

        zy = z if y is None else torch.cat((z, y), dim=1)
        # print(zy.shape)
        zy1, zy2, zy3, zy4 = torch.split(zy, self.z_dim // self.concept, dim=1)  # each is 16x16
        # print(zy1.shape)
        # sys.exit(0)
        rx1 = self.net1.decode(zy1)
        # print(rx1.shape)
        # sys.exit(0)
        # print(f"Hi: {rx1.size()}")
        rx2 = self.net2.decode(zy2)
        rx3 = self.net3.decode(zy3)
        rx4 = self.net4.decode(zy4)
        # z = torch.cat((rx1, rx2, rx3, rx4), dim=0)
        z = (rx1+rx2+rx3+rx4)/4
        # print(z.shape)
        # sys.exit(0)
        return z, z, z, z, z

    def decode(self, z, u=None, y=None):
        z = z.view(-1, self.concept * self.z1_dim, 1, 1)
        z = self.net6(z)
        print(z.size())

        return z
