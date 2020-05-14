from torch import nn, optim
from ..utils.progress import Progress



class Module(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        
        # call `__init__` of `nn.Module`
        super(Module, instance).__init__()
        
        instance.optim = optim.Adam
        instance.lr = 1e-3
        return instance
    
    def __init__(self):
        pass
    

    def train(self, loader, epoch = 10):
        """train model
        """
        optimizer = self.optim(self.parameters(), lr = self.lr)
        for epoch in range(10):
            p = Progress(loader)
            p.prefix = f"Epoch:{epoch}"

            for x, y in p:
                y_hat = self.__call__(x)
                loss = self.calculate_loss(y_hat, y, x)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                p.suffix = 'loss:{:.4f}'.format(loss.item())
    
    def calculate_loss(self, y_hat, y, x):
        """calculate loss
        """
        return nn.functional.mse_loss(y_hat, y)