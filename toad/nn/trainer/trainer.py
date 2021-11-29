import torch
import numpy as np
from torch import optim

from .history import History
from .callback import callback as Callback

from ...utils.progress import Progress


        
class Trainer:
    """trainer for training models
    """
    def __init__(self, model, loader = None, optimizer = None, loss = None, keep_history = None,
                 early_stopping = None):
        """initialization

        Args:
            model (nn.Module): model will be trained
            loader (torch.DataLoader): training data loader
            optimizer (torch.Optimier): the default optimizer is `Adam(lr = 1e-3)`
            loss (Callable): could be called as 'loss(y_hat, y)'
            early_stopping (earlystopping): the default value is `loss_earlystopping`, 
                you can set it to `False` to disable early stopping
            keep_history (int): keep the last n-th epoch logs, `None` will keep all
        """
        self.model = model
        self.loader = loader

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
        
        self.optimizer = optimizer

        self.loss = loss

        # set default early stopping
        if early_stopping is None:
            from .earlystop import loss_stopping
            early_stopping = loss_stopping()
        
        self.early_stop = early_stopping

        from collections import deque
        self.history = deque(maxlen = keep_history)


    def train(self, loader = None, epoch = 10, callback = [], start = 0, backward_rounds = 1):
        """
        Args:
            loader (torch.DataLoader): training data loader
            epoch (int): number of epoch for training loop
            callback (callable): callable function will be called every epoch
                - parameters of callback
                    model (nn.Module): the training model
                    history (History): history of total log records
                    epoch (int): current epoch number
                    trainer (Trainer): self trainer
            start (int): epoch start from n round
            backward_rounds (int): backward after every n rounds 
        
        Returns:
            Module: the model with best performents
        """
        if loader is not None:
            self.loader = loader
        
        if self.loader is None:
            raise ValueError("loader is not set, please set a loader for trainning!")

        if callable(callback) and not isinstance(callback, Callback):
            callback = Callback(callback)
        
        if not isinstance(callback, list):
            callback = [callback]
        
        # init progress bar
        p = Progress(self.loader)

        for ep in range(start, epoch):
            # set model to train mode
            self.model.train()

            p.prefix = f"Epoch:{ep}"

            # setup a new history for model in each epoch
            history = History()
            self.history.append(history)
            self.model._history = history

            loss = 0.
            backward_loss = 0.
            for i, batch in enumerate(p, start = 1):
                # step fit
                if self.loss is None:
                    l = self.model.fit_step(batch)
                else:
                    l = self.model.fit_step(batch, loss=self.loss)

                # log loss
                self.model.log('loss', l)
                
                backward_loss = l + backward_loss
                if i % backward_rounds == 0 or i == len(p):
                    self.optimizer.zero_grad()
                    backward_loss.backward()
                    self.optimizer.step()
                    
                    # reset backward loss
                    backward_loss = 0.

                loss += (l.item() - loss) / i
                p.suffix = 'loss:{:.4f}'.format(loss)

            # setup callback params
            callback_params = {
                "model": self.model,
                "history": history,
                "epoch": ep,
                "trainer": self,
            }

            with torch.no_grad():
                if isinstance(callback, list):
                    for hook in callback:
                        hook(**callback_params)
                
                if self.early_stop and self.early_stop(**callback_params):
                    # set best state to model
                    best_state = self.early_stop.get_best_state()
                    self.model.load_state_dict(best_state)
                    break
        
        return self.model
    

    @torch.no_grad()
    def evaluate(self, loader, callback = None):
        """evalute model

        Args:
            loader (torch.DataLoader): evaluation data loader
            callback (callable): callback function
        """
        if callback and not isinstance(callback, Callback):
            callback = Callback(callback)
        
        # init progress bar
        p = Progress(loader)
        p.prefix = f"Evaluate"

        history = History()
        self.model._history = history

        self.model.eval()
        
        loss = 0.
        for i, batch in enumerate(p, start = 1):
            # step fit
            if self.loss is None:
                l = self.model.fit_step(batch)
            else:
                l = self.model.fit_step(batch, loss=self.loss)

            # log loss
            self.model.log('loss', l)

            loss += (l.item() - loss) / i
            p.suffix = 'loss:{:.4f}'.format(loss)
        
        if callable(callback):
            callback(
                epoch = None,
                history = history,
                trainer = self,
                model = self.model,
            )
        
        return history
