from torch import nn, optim
from ..utils.progress import Progress





class AutoEncoder(nn.Module):
    def __init__(self, input, hidden, zipped):
        super().__init__()

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
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)

    def forward(self, x):
        z = self.encode(x)
        return self.decode(z)
    
    def fit(self, X):
        loss = nn.MSELoss()
        optimizer = optim.Adam(self.parameters(), lr = 1e-3)
        for epoch in range(10):
            p = Progress(X)
            p.prefix = epoch

            for (x, _) in p:
                x = x.view(-1, 784)
                x_hat = self.forward(x)
                l = loss(x_hat, x)
                
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                p.suffix = '{:.4f}'.format(l.item())
                # print(epoch, l.item())




