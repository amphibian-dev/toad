import torch
import numpy as np
from .history import History

def test_history_log():
    history = History()

    for i in range(10):
        history.log('tensor', torch.rand(3, 5))
    
    assert history['tensor'].shape == (30, 5)


