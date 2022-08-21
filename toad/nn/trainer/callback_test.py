from torch import nn

from .callback import callback, savemodel
from ..module import Module

class TestModel(Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        self.linear = nn.Linear(in_feats, out_feats)


def test_callback():
    @callback
    def hook(history, trainer):
        return history['a']
    
    res = hook(epoch = 1, trainer = None, history = {"a": 3})

    assert res == 3

def test_checkpoint():
    model = TestModel(10, 2)
    hook = savemodel(dirpath = '/dev', filename = "null")
    hook(model = model, epoch = 1)
