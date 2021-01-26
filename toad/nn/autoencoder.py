import torch
from torch import nn, optim
from torch.nn.functional import relu, binary_cross_entropy

from .module import Module
from ..utils.progress import Progress



class BaseAutoEncoder(Module):
    def __init__(self, input, hidden, zipped):

        self.encoder = nn.Sequential(
            nn.Linear(input, hidden),
            nn.ReLU(),
            nn.Linear(hidden, zipped),
        )

        self.decoder = nn.Sequential(
            nn.Linear(zipped, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input),
        )

        self.loss = nn.MSELoss()
    
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def fit_step(self, x):
        return self.loss(self(x), x)



class VAE(Module):
    def __init__(self, input, hidden, zipped):
        self.hidden_layer = nn.Linear(input, hidden)

        self.mu_layer = nn.Linear(hidden, zipped)
        self.var_layer = nn.Linear(hidden, zipped)

        self.decoder = nn.Sequential(
            nn.Linear(zipped, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input),
        )

        self.loss = nn.MSELoss()
    
    def encode(self, x):
        h = relu(self.hidden_layer(x))
        mu = self.mu_layer(h)
        var = self.var_layer(h)

        std = torch.exp(var / 2)
        eps = torch.rand_like(std)
        
        z = mu + eps * std
        return z, mu, var
    
    def decode(self, x):
        return self.decoder(x)
    
    def forward(self, x):
        z, mu, var = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, mu, var
    
    def fit_step(self, x):
        x_hat, mu, var = self(x)
        l = self.loss(x_hat, x)
        kld = -0.5 * torch.sum(1 + var - torch.pow(mu, 2) - torch.exp(var))

        loss = l + kld
        return loss
