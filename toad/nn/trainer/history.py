import torch
import numpy as np


_history_stack = [None]


def get_current_history():
    global _history_stack

    return _history_stack[-1]



class History:
    """model history
    """
    def __init__(self):
        self._store = {}
    

    def __getitem__(self, key):
        return self._store[key]
    

    def __setitem__(self, key, value):
        return self.log(key, value)
    

    def _push(self, key, value):
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
    

    def log(self, key, value):
        """log message to history

        Args:
            key (str): name of message
            value (Tensor): tensor of values
        """
        if isinstance(value, torch.Tensor):
            value = value.detach().cpu().numpy()
            
            # fix scaler tensor
            if value.ndim == 0:
                value = value.reshape(-1)

        if np.isscalar(value):
            value = np.array([value])
        
        if not isinstance(value, np.ndarray):
            raise TypeError("value should be `torch.Tensor` or `scalar`")
        
        self._push(key, value)
    

    def start(self):
        global _history_stack
        _history_stack.append(self)
        
        return self
    

    def end(self):
        global _history_stack
        return _history_stack.pop()
    

    def __enter__(self):
        return self.start()
    

    def __exit__(self, exc_type, exc_val, exc_tb):
        return self.end()
