import numpy as np
from ..utils.progress import Progress

class EarlyStopping:
    def __init__(self, delta = -1e-3, patience = 10, skip = 0):
        """
        Args:
            delta (float): the smallest change that considered as an improvement.
                If is positive, means larger is better, negative means smaller is better.
            patience (int): how many times will be stop when score has no improvement
            skip (int): how many rounds should skip after start training
        """
        self.direction = 1.0 if delta > 0 else -1.0
        self.delta = delta * self.direction
        self.patience = patience
        self.skip = skip
        
        self.reset()

    
    def get_best_state(self):
        """get best state of model
        """
        return self.best_state
    

    def reset(self):
        """
        """
        self.best_score = float('inf') * (-self.direction)
        self.best_state = None
        self._times = 0
        self._round = -1
    

    def __call__(self, model, *args, **kwargs):
        self._round += 1

        # set skip round
        if self._round < self.skip:
            return False
        
        score = self.scoring(model, *args, **kwargs)
        diff = (score - self.best_score) * self.direction
        
        if diff > self.delta:
            self.best_state = model.state_dict()
            self.best_score = score
            self._times = 0
            return False
        
        self._times += 1
        if self._times >= self.patience:
            # model.load_state_dict(self.best_state)
            return True
    
    def scoring(self, model, loss, epoch = None):
        """scoring function
        """
        return loss
        


class History:
    """model history
    """
    def __init__(self):
        self._store = {}
    

    def __getitem__(self, key):
        return self._store[key]
    

    def __setitem__(self, key, value):
        return self.push(key, value)
    
    def push(self, key, value):
        """push value into history

        Args:
            key (str): key of history
            value (np.ndarray): an array of values
        """
        if key not in self._store:
            self._store[key] = value
            return

        self._store[key] = np.concatenate([
            self._store[key],
            value,
        ])


        
class Trainer:
    def __init__(self, model, loader, optimizer = None, early_stopping = None):
        self.model = model
        self.loader = loader

        if optimizer is None:
            optimizer = model.optimizer()
        self.optimizer = optimizer

        self.early_stop = early_stopping
        self.history = []


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


            if self.early_stop and self.early_stop(self.model, loss, epoch = ep):
                # set best state to model
                best_state = self.early_stop.get_best_state()
                self.model.load_state_dict(best_state)
                break
            
            if callable(callback):
                callback(ep, history)
        
        return self.model
