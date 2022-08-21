import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from .module import Module
from .loss import DictLoss, ListLoss

DATASET_SIZE = 20000
NUM_FEATS = 784
NUM_CLASSES = 2

X = torch.rand(DATASET_SIZE, NUM_FEATS, dtype=torch.float)
y = torch.randint(NUM_CLASSES, size=(DATASET_SIZE,), dtype=torch.long)


class DictDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], {'y': self.y[item]}


class ListDataset(Dataset):
    def __init__(self, x, y):
        super().__init__()
        self.x = x
        self.y = y

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, item):
        return self.x[item], [self.y[item]]


class TestDictModel(Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        x = self.linear(x)
        return {'y': F.relu(x)}

    def fit_step(self, batch, loss=None):
        x, y = batch
        y_hat = self(x)
        return loss(y_hat, y)


class TestListModel(Module):
    def __init__(self, in_feats, out_feats):
        super().__init__()

        self.linear = nn.Linear(in_feats, out_feats)

    def forward(self, x):
        x = self.linear(x)
        return [F.relu(x)]

    def fit_step(self, batch, loss=None):
        x, y = batch
        y_hat = self(x)
        return loss(y_hat, y)


def test_dict_loss():
    model = TestDictModel(NUM_FEATS, NUM_CLASSES)
    loader = DataLoader(
        DictDataset(X, y),
        batch_size=128,
        shuffle=True,
    )
    model.fit(loader, epoch=1, loss=DictLoss(F.cross_entropy))


def test_list_loss():
    model = TestListModel(NUM_FEATS, NUM_CLASSES)
    loader = DataLoader(
        ListDataset(X, y),
        batch_size=128,
        shuffle=True,
    )
    model.fit(loader, epoch=1, loss=ListLoss(F.cross_entropy))
