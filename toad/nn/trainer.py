from ..utils.progress import Progress

class EarlyStopping:
    def __init__(self, delta = -1e-3, patience = 10):
        """
        Args:
            delta (float): the smallest change that considered as an improvement.
                If is positive, means larger is better, negative means smaller is better.
            patience (int): how many times will be stop when score has no improvement
        """
        self.direction = 1.0 if delta > 0 else -1.0
        self.delta = delta * self.direction
        self.patience = patience
        
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
    

    def __call__(self, model, *args, **kwargs):
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
    
    def scoring(self, model, loss):
        """scoring function
        """
        return loss
        

        
class Trainer:
    def __init__(self, model, loader, optimizer = None, early_stopping = None):
        self.model = model
        self.loader = loader

        if optimizer is None:
            optimizer = model.optimizer()
        self.optimizer = optimizer

        self.early_stop = early_stopping
        self.loss_history = []

    def train(self, epoch = 10, callback = None):
        # init progress bar
        p = Progress(self.loader)

        for ep in range(epoch):
            p.prefix = f"Epoch:{ep}"

            loss = 0.
            for i, batch in enumerate(p, start = 1):
                # step fit
                l = self.model.fit_step(batch)
                
                self.optimizer.zero_grad()
                l.backward()
                self.optimizer.step()

                loss += (l.item() - loss) / i
                p.suffix = 'loss:{:.4f}'.format(loss)

            self.loss_history.append(loss)

            if self.early_stop and self.early_stop(self.model, loss):
                # set best state to model
                best_state = self.early_stop.get_best_state()
                self.model.load_state_dict(best_state)
                break
            
            if callable(callback):
                callback(ep, loss)
        
        return self.model
