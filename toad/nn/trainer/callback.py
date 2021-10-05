from ...utils.decorator import Decorator

class callback(Decorator):
    """
    Examples:
        >>> @callback
        ... def savemodel(model):
        ...     model.save("path_to_file")
        ...
        ... trainer.train(model, callback = savemodel)
    """
    def setup_func(self, func):
        import inspect
        self._params = inspect.signature(func).parameters
        
        return func
        

    def wrapper(self, **kwargs):
        params = {k: v for k ,v in kwargs.items() if k in self._params.keys()}

        return self.call(**params)
