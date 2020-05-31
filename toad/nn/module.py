from torch import nn, optim
from ..utils.progress import Progress



class Module(nn.Module):
    def __new__(cls, *args, **kwargs):
        instance = super().__new__(cls)
        
        # call `__init__` of `nn.Module`
        super(Module, instance).__init__()

        return instance
    
    def __init__(self):
        pass
    

    def fit(self, loader, epoch = 10):
        """train model
        """
        optimizer = self.optimizer()

        for ep in range(epoch):
            p = Progress(loader)
            p.prefix = f"Epoch:{ep}"

            for x, y in p:
                loss = self.fit_step(x, y)
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                p.suffix = 'loss:{:.4f}'.format(loss.item())
    
    def fit_step(self, x, y):
        """step for fitting
        Args:
            x (Tensor)
            y (Tensor)
        
        Returns:
            Tensor: loss of this step
        """
        y_hat = self.__call__(x)
        loss = nn.functional.mse_loss(y_hat, y)
        return loss

    def optimizer(self):
        return optim.Adam(self.parameters(), lr = 1e-3)
    