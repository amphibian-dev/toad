from toad.nn.trainer.history import History
import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from ..module import Module
from .trainer import Trainer
from .callback import callback
from .earlystop import earlystopping


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
        super().__init__()

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
    assert len(trainer.history) == 2


def test_trainer_early_stopping():
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    
    @earlystopping(delta = -1.0, patience = 3)
    def scoring(history):
        return history['loss'].mean()

    trainer = Trainer(model, loader, early_stopping = scoring)
    trainer.train(epoch = 200)
    assert len(trainer.history) == 4


def test_multi_callbacks():
    log = {}
    
    @callback
    def log_epoch(epoch):
        log['epoch'] = epoch
    
    @callback
    def log_loss(history):
        log['loss'] = history['loss']

    model = TestModel(NUM_FEATS, NUM_CLASSES)
    trainer = Trainer(model)
    trainer.train(loader, epoch = 2, callback = [log_epoch, log_loss])
    
    assert log['epoch'] == 1
    assert len(log['loss']) == 157


class TestModel2(TestModel):
    def fit_step(self, batch, loss=None):
        x, y = batch
        y_hat = self(x)
        return loss(y_hat, y)


def test_trainer_loss():
    model = TestModel2(NUM_FEATS, NUM_CLASSES)
    trainer = Trainer(model, loader, loss = F.cross_entropy)
    trainer.train(epoch = 2)
    assert len(trainer.history) == 2
