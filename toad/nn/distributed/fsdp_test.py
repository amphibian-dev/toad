from ..module import Module
from .fsdp import FSDPModule

import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader

from toad.utils.progress import Progress



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
        # return F.cross_entropy(y_hat, y)
        return F.mse_loss(y_hat, y)

def worker(rank, world):
    from torch.distributed.fsdp.wrap import (
        size_based_auto_wrap_policy,
        ModuleWrapPolicy,
    )

    torch.manual_seed(0)

    NUM_FEATS = 4096
    NUM_CLASSES = 1024
    DATASET_SIZE = 10000


    X = torch.rand(DATASET_SIZE, NUM_FEATS, dtype = torch.float)
    # y = torch.randint(NUM_CLASSES, size = (DATASET_SIZE,), dtype = torch.long)

    NUM_CLASSES = 1
    y = torch.sum(X, dim = 1)

    loader = DataLoader(
        TensorDataset(X, y),
        batch_size = 128,
        shuffle = True,
    )

    model = TestModel(NUM_FEATS, NUM_CLASSES)
    # print(next(model.linear.parameters()).shape)

    model.distributed(backend = "gloo", rank = rank, world_size = world)
    
    fdsp_model = FSDPModule(
        model,
        # sync_module_states = True,
        # auto_wrap_policy = my_auto_wrap_policy,
        # policy = ModuleWrapPolicy([nn.Linear,]),
        device_id=torch.device("cpu"),
    )

    optimizer = optim.Adam(fdsp_model.parameters(), lr = 1e-3)

    state_path = f"data/fsdp_model_{rank}.pkl"

    fdsp_model.load(state_path)

    print('before fit:', fdsp_model(X[0]).sum())

    # inputs = torch.rand(10, features_dim)
    fdsp_model.fit(loader, epoch = 20, early_stopping = False)

    print('after fit:', fdsp_model(X[0]).sum())

    print(fdsp_model)
    # print(fdsp_model.flatten_sharded_optim_state_dict())

    # out = fdsp_model(inputs).sum()

    # out.backward()

    # print("~~~~~", out)

    print(model)
    model.save(f"data/origin_model_{rank}.pkl")
    
    fdsp_model.save(state_path)




def test_fsdp_model():
    import torch.multiprocessing as mp
    
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(worker, args=(2,), nprocs=2, join=True)

