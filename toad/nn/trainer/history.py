import torch
import numpy as np

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
