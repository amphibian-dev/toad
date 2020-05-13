from torch import nn, optim





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
            for (x, _) in X:
                x = x.view(-1, 784)
                x_hat = self.forward(x)
                l = loss(x_hat, x)
                
                optimizer.zero_grad()
                l.backward()
                optimizer.step()

                print(epoch, l.item())




