from torch import nn, optim
from .module import Module
from ..utils.progress import Progress



class BaseAutoEncoder(Module):
    def __init__(self, input, hidden, zipped):
        # super().__init__()

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
        self.optim = optim.Adam
        self.lr = 1e-3
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def calculate_loss(self, y_hat, y, x):
        return self.loss(y_hat, x)
    
    def fit(self, X):
        # TODO convert X to loader
        self.train(X)




