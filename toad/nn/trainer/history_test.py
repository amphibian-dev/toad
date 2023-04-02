import torch
import numpy as np
from .history import History, get_current_history

def test_history_log():
    history = History()

    for i in range(10):
        history.log('tensor', torch.rand(3, 5))
    
    assert history['tensor'].shape == (30, 5)


def test_current_history():
    history = History()

    with history:
        h = get_current_history()
        h.log('tensor', torch.rand(3, 5))
    
    assert history['tensor'].shape == (3, 5)
