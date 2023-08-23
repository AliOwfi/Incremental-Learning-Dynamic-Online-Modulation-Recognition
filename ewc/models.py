import torch 
import torch.nn as nn
import numpy as np


class MLP(nn.Module):
    def __init__(self, arch, zero_init=False, sig=False, act='relu'):
        super(MLP, self).__init__()
        
        self.sig = sig
        self.arch = np.array(arch)

        self.layers = []
        for l_cnt, (in_dim, out_dim) in enumerate(zip(arch[:-1], arch[1:])):
            
            self.layers.append(nn.Linear(in_dim, out_dim))
        
            if zero_init == True:
                torch.nn.init.zeros_(self.layers[-1].weight)
            
        self.layers = nn.ModuleList(self.layers)            

        if act == 'relu':
            self.nonlin = nn.ReLU()
        elif act == 'tanh':
            self.nonlin = nn.Tanh()
        elif act == 'soft':
            self.nonlin = nn.Softplus() 

    def reset_params(self):
        for layer in self.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()

    def forward(self, x):

        x = x.reshape(x.shape[0], -1)  
        for l_idx, l in enumerate(self.layers):
            x = l(x)
            if l_idx < (len(self.layers) - 1):
                x = self.nonlin(x)

        if self.sig == True:
            x = torch.sigmoid(x)

        return x 
