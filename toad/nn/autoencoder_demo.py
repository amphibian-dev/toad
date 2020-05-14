import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .autoencoder import BaseAutoEncoder


X = torch.Tensor(np.random.rand(20000, 784))
Y = torch.Tensor(np.random.randint(10, size = (20000, 1)))

dataset = TensorDataset(X, Y)
loader = DataLoader(
    dataset,
    batch_size = 128,
    shuffle = True,
)

ae = BaseAutoEncoder(784, 200, 10)
ae.fit(loader)