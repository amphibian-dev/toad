from torch import nn, optim
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
    
    def fit_step(self, x, y):
        return self.loss(self(x), x)



