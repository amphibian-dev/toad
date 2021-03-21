import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .module import Module
from .trainer import Trainer, EarlyStopping


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


def test_trainer():
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    trainer = Trainer(model, loader)
    trainer.train(epoch = 2)
    assert len(trainer.loss_history) == 2


def test_trainer_early_stopping():
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    early_stopping = EarlyStopping(delta = -1.0, patience = 3)
    trainer = Trainer(model, loader, early_stopping = early_stopping)
    trainer.train(epoch = 200)
    assert len(trainer.loss_history) == 4
