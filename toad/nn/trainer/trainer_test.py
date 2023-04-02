import pytest

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from .history import History
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


def test_trainer_fit_step():
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    trainer = Trainer(model, loader)
    step_count = 0

    @trainer.fit_step
    def step(model, batch):
        x, y = batch
        y_hat = model(x)
        nonlocal step_count
        step_count += 1
        return F.cross_entropy(y_hat, y)
    
    trainer.train(epoch = 2)
    assert step_count > 1


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


def test_trainer_evaluate():
    model = TestModel(NUM_FEATS, NUM_CLASSES)
    trainer = Trainer(model, loader)

    @trainer.fit_step
    def step(model, batch):
        x, y = batch
        y_hat = model(x)
        return F.cross_entropy(y_hat, y)
    
    history = trainer.evaluate(loader)

    assert len(history["loss"]) == 157
    


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


# def test_trainer_distributed():
#     model = TestModel(NUM_FEATS, NUM_CLASSES)
#     trainer = Trainer(model, loader)
#     trainer.distributed(workers = 2)
#     trainer.train(epoch = 5)



### distribut model test
# from toad.nn.trainer.trainer import Trainer
# from torchvision.transforms import ToTensor
# import torch
# from torch import nn
# from torchvision import datasets
# from toad.nn import Module
# from torch.utils.data import DataLoader
# import ray

# class NeuralNetwork(Module):
#     def __init__(self):
#         super(NeuralNetwork, self).__init__()
#         self.flatten = nn.Flatten()
#         self.linear_relu_stack = nn.Sequential(
#             nn.Linear(28 * 28, 512),
#             nn.ReLU(),
#             nn.Linear(512, 512),
#             nn.ReLU(),
#             nn.Linear(512, 10),
#             nn.ReLU(),
#         )
#     def forward(self, x):
#         x = self.flatten(x)
#         logits = self.linear_relu_stack(x)
#         return logits
#     def fit_step(self, batch):
#         X, y = batch
#         pred =self(X)
#         loss_fn=nn.CrossEntropyLoss()
#         return loss_fn(pred, y)


# @pytest.mark.skip("distributed trainer skip")
# def test_distribute_example():
#     training_data = datasets.FashionMNIST(
#         root="~/data",
#         train=True,
#         download=True,
#         transform=ToTensor(),
#     )
#     # Download test data from open datasets.
#     test_data = datasets.FashionMNIST(
#         root="~/data",
#         train=False,
#         download=True,
#         transform=ToTensor(),
#     )
#     worker_batch_size = 64 // 4
#     # Create data loaders.
#     train_dataloader = DataLoader(training_data, batch_size=16)
#     test_dataloader = DataLoader(test_data, batch_size=16)
#     model=NeuralNetwork()
#     optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
#     trainer=Trainer(model,train_dataloader,optimizer)
#     trainer.distributed(address="ray://172.20.144.21:10001",num_works=4,use_gpu=False)
#     trainer.train(epoch=1)
#     trainer.evaluate(test_dataloader)
