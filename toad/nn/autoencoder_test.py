import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from .autoencoder import BaseAutoEncoder, VAE


X = torch.Tensor(np.random.rand(20000, 784))

loader = DataLoader(
    X,
    batch_size = 128,
    shuffle = True,
)

def test_ae():
    ae = BaseAutoEncoder(784, 200, 10)
    ae.fit(loader, epoch = 1)

def test_vae():
    vae = VAE(784, 200, 10)
    vae.fit(loader, epoch = 1)
