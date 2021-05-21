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
    
    def scoring(self, model, history, epoch = None):
        """scoring function
        """
        return history['loss'].mean()
