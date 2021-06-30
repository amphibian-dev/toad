import numpy as np
from torch import optim

from .history import History
from .earlystopping import EarlyStopping
from ...utils.progress import Progress
        

        
class Trainer:
    def __init__(self, model, loader, optimizer = None, keep_history = None,
                early_stopping = EarlyStopping()):
        """
        Args:
            model (nn.Module)
            loader (torch.DataLoader)
            optimizer (torch.Optimier)
            early_stopping (EarlyStopping)
            keep_history (int): keep the last n-th epoch logs, `None` will keep all
        """
        self.model = model
        self.loader = loader

        if optimizer is None:
            optimizer = optim.Adam(model.parameters(), lr = 1e-3)
        
        self.optimizer = optimizer

        self.early_stop = early_stopping

        from collections import deque
        self.history = deque(maxlen = keep_history)


    def train(self, epoch = 10, callback = None):
        """
        Args:
            epoch (int): number of epoch for training loop
            callback (callable): callable function will be called every epoch
        """
        # init progress bar
        p = Progress(self.loader)

        

        for ep in range(epoch):
            p.prefix = f"Epoch:{ep}"

            # setup a new history for model in each epoch
            history = History()
            self.history.append(history)
            self.model._history = history

            loss = 0.
            for i, batch in enumerate(p, start = 1):
                # step fit
                l = self.model.fit_step(batch)

                # log loss
                self.model.log('loss', l)
                
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                loss += (l.item() - loss) / i
                p.suffix = 'loss:{:.4f}'.format(loss)


            if self.early_stop and self.early_stop(self.model, history, epoch = ep):
                # set best state to model
                best_state = self.early_stop.get_best_state()
                self.model.load_state_dict(best_state)
                break
            
            if callable(callback):
                callback(ep, history)
        
        return self.model
