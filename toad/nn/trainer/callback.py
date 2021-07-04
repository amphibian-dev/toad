from ...utils.decorator import Decorator

class callback(Decorator):
    def setup_func(self, func):
        import inspect
        self._params = inspect.signature(func).parameters
        
        return func
        

    def wrapper(self, **kwargs):
        params = {k: v for k ,v in kwargs.items() if k in self._params.keys()}

        return self.call(**params)
