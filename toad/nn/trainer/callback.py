from ...utils.decorator import Decorator

class callback(Decorator):
    """callback for trainer
    
    Examples:
        >>> @callback
        ... def savemodel(model):
        ...     model.save("path_to_file")
        ...
        ... trainer.train(model, callback = savemodel)
    
    """
    def __init__(self, *args, **kwargs):
        if hasattr(self, 'wrapped'):
            # use `wrapped` func as core func
            super().__init__(getattr(self, 'wrapped'))
            # setup configuration
            self.setup(*args, **kwargs)
            return
        
        # init normal decorator
        super().__init__(*args, **kwargs)


    def setup_func(self, func):
        import inspect
        self._params = inspect.signature(func).parameters
        
        return func
        

    def wrapper(self, **kwargs):
        params = {k: v for k ,v in kwargs.items() if k in self._params.keys()}

        return self.call(**params)



class checkpoint(callback):
    """
    Args:
        dir (string): dir name for saving checkpoint
        every (int): every epoch for saving
        format (string): checkpoint file format
    """
    dirpath = "model_checkpoints"
    every = 1
    filename = "{name}-{epoch}.pt"
    

    def wrapper(self, **kwargs):
        model = kwargs.get("model")
        epoch = kwargs.get("epoch")

        name = type(model).__name__

        from pathlib import Path
        dirpath = Path(self.dirpath)
        dirpath.mkdir(parents = True, exist_ok = True)

        filename = self.filename.format(
            name = name,
            epoch = epoch,
        )

        path = dirpath / filename

        if epoch % self.every == 0:
            super().wrapper(
                path = path,
                **kwargs
            )


class savemodel(checkpoint):
    """
    Args:
        dir (string): dir name for saving checkpoint
        every (int): every epoch for saving
        format (string): checkpoint file format, default is `{name}-{epoch}.pt`
    """
    def wrapped(self, model, path):
        import torch
        torch.save(model.state_dict(), path)
