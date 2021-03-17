import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .module import Module

DATASET_SIZE = 20000
NUM_FEATS = 784
NUM_CLASSES = 2

X = torch.rand(DATASET_SIZE, NUM_FEATS, dtype = torch.float)
y = torch.randint(NUM_CLASSES, size = (DATASET_SIZE,), dtype = torch.long)

loader = DataLoader(
    TensorDataset(X, y),
    batch_size = 128,
    shuffle = True,
)

class TestModel(Module):
    def __init__(self, in_feats, out_feats):
        self.linear = nn.Linear(in_feats, out_feats)
    
    def forward(self, x):
        x = self.linear(x)
        return F.relu(x)
    
    def fit_step(self, batch):
        x, y = batch
        y_hat = self(x)
        return F.cross_entropy(y_hat, y)

def test_model():
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    model.fit(loader, epoch = 1)


def test_fit_callback():
    history = []

    def func(e, loss):
        history.append(loss)
    
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    model.fit(loader, epoch = 2, callback = func)
    assert len(history) == 2
