from .callback import callback
from ...utils.decorator import Decorator


class earlystopping(Decorator):
    """
    Examples:
        >>> @earlystopping(delta = 1e-3, patience = 5)
        ... def auc(history):
        ...     return AUC(history['y_hat'], history['y'])
    """
    def __init__(self, *args, delta = -1e-3, patience = 10, skip = 0, **kwargs):
        """
        Args:
            delta (float): stop training if diff of new score is smaller than delta
            patience (int): patience of rounds to stop training
            skip (int): n rounds from starting training to warm up
        """
        super().__init__(*args, **kwargs)
        
        self.direction = 1.0 if delta > 0 else -1.0
        self.delta = delta * self.direction
        self.patience = patience
        self.skip = skip
        
        self.reset()
    

    def setup_func(self, func):
        if not isinstance(func, callback):
            func = callback(func)
        
        return func
    
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
    

    def wrapper(self, model, epoch = 0, **kwargs):
        # set skip round
        if epoch < self.skip:
            return False
        
        score = self.call(model = model, epoch = epoch, **kwargs)
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
        

@earlystopping
def loss_scoring(history):
    """scoring function
    """
    return history['loss'].mean()
