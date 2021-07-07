from toad.nn import trainer
import numpy as np
from torch import optim

from .history import History
from .callback import callback as Callback
from .earlystop import earlystopping

from ...utils.progress import Progress



@earlystopping
def loss_scoring(history):
    """scoring function
    """
    return history['loss'].mean()


        
class Trainer:
    def __init__(self, model, loader, optimizer = None, keep_history = None,
                early_stopping = loss_scoring):
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


    def train(self, epoch = 10, callback = None, start = 0, backward_rounds = 1):
        """
        Args:
            epoch (int): number of epoch for training loop
            callback (callable): callable function will be called every epoch
            backward_rounds (int): backward after every n rounds 
        """
        if callback and not isinstance(callback, Callback):
            callback = Callback(callback)
        
        # init progress bar
        p = Progress(self.loader)

        self.model.train()

        for ep in range(start, epoch):
            p.prefix = f"Epoch:{ep}"

            # setup a new history for model in each epoch
            history = History()
            self.history.append(history)
            self.model._history = history

            loss = 0.
            backward_loss = 0.
            for i, batch in enumerate(p, start = 1):
                # step fit
                l = self.model.fit_step(batch)

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

            if self.early_stop and self.early_stop(**callback_params):
                # set best state to model
                best_state = self.early_stop.get_best_state()
                self.model.load_state_dict(best_state)
                break
            
            if callable(callback):
                callback(**callback_params)
        
        return self.model
    

    def evaluate(self, loader, callback = None):
        """evalute model
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
            l = self.model.fit_step(batch)

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
