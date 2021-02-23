import sys
import torch
import pytest
import numpy as np
from torch.utils.data import TensorDataset, DataLoader

from .autoencoder import BaseAutoEncoder, VAE

# skip testing with python 3.9 on linux
if sys.version_info >= (3, 9) and sys.platform.startswith('linux'):
    pytest.skip("failed with python 3.9 on linux, need fix!", allow_module_level = True)


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
