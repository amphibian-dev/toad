import torch
import numpy as np
from torch import nn, optim
from torch.nn.parallel import DistributedDataParallel

from .trainer.history import get_current_history
from ..utils.progress import Progress



class Module(nn.Module):
    """base module for every model

    Examples:
        >>> from toad.nn import Module
        ... from torch import nn
        ... 
        ... class Net(Module):
        ...     def __init__(self, inputs, hidden, outputs):
        ...         super().__init__()
        ...         self.model = nn.Sequential(
        ...             nn.Linear(inputs, hidden),
        ...             nn.ReLU(),
        ...             nn.Linear(hidden, outputs),
        ...             nn.Sigmoid(),
        ...         )
        ...     
        ...     def forward(self, x):
        ...         return self.model(x)
        ...     
        ...     def fit_step(self, batch):
        ...         x, y = batch
        ...         y_hat = self(x)
        ... 
        ...         # log into history
        ...         self.log('y', y)
        ...         self.log('y_hat', y_hat)
        ... 
        ...         return nn.functional.mse_loss(y_hat, y)
        ... 
        ... model = Net(10, 4, 1)
        ... 
        ... model.fit(train_loader)

    """
    def __init__(self):
        """define model struct
        """
        super().__init__()
    

    @property
    def device(self):
        """device of model
        """
        return next(self.parameters()).device


    def fit(self, loader, trainer = None, optimizer = None, loss = None, early_stopping = None, **kwargs):
        """train model

        Args:
            loader (DataLoader): loader for training model
            trainer (Trainer): trainer for training model
            optimizer (torch.Optimier): the default optimizer is `Adam(lr = 1e-3)`
            loss (Callable): could be called as 'loss(y_hat, y)'
            early_stopping (earlystopping): the default value is `loss_earlystopping`, 
                you can set it to `False` to disable early stopping
            epoch (int): number of epoch for training loop
            callback (callable): callable function will be called every epoch
        """
        if trainer is None:
            from .trainer import Trainer
            trainer = Trainer(self, loader, optimizer = optimizer, loss = loss, early_stopping = early_stopping)
        
        trainer.train(**kwargs)
    

    def evaluate(self, loader, trainer = None):
        """evaluate model
        
        Args:
            loader (DataLoader): loader for evaluate model
            trainer (Trainer): trainer for evaluate model
        """
        if trainer is None:
            from .trainer import Trainer
            trainer = Trainer(self)
        
        return trainer.evaluate(loader)

    

    def fit_step(self, batch, loss = None, *args, **kwargs):
        """step for fitting
        
        Args:
            batch (Any): batch data from dataloader
            loss (Callable): could be called as 'loss(y_hat, y)'
        
        Returns:
            Tensor: loss of this step
        """
        x, y = batch
        y_hat = self.__call__(x)
        if loss is None:
            loss = nn.functional.mse_loss
        return loss(y_hat, y)


    def save(self, path):
        """save model
        """
        torch.save(self.state_dict(), path)
    

    def load(self, path):
        """load model
        """
        state = torch.load(path)
        self.load_state_dict(state)
    

    def log(self, key, value):
        """log values to history

        Args:
            key (str): name of message
            value (Tensor): tensor of values
        """
        history = get_current_history()
        if history is None:
            return
        
        return history.log(key, value)
        
        
    def distributed(self, backend = None, **kwargs):
        """get distributed model
        """
        if not torch.distributed.is_initialized():
            if backend is None:
                # choose a backend
                backend = 'nccl' if torch.distributed.is_nccl_available() else 'gloo'

            torch.distributed.init_process_group(backend, **kwargs)
        
        return DistModule(self)
        


class DistModule(DistributedDataParallel):
    """distributed module class
    """
    def fit(self, *args, **kwargs):
        return self.module.fit(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        return self.module.save(*args, **kwargs)
    
    def load(self, *args, **kwargs):
        return self.module.load(*args, **kwargs)
    
    def log(self, *args, **kwargs):
        return self.module.log(*args, **kwargs)
    
    
