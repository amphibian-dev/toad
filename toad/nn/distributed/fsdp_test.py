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
        
        self.linear_1 = nn.Linear(in_feats, in_feats)
        self.linear_2 = nn.Linear(in_feats, out_feats)
    
    def forward(self, x):
        x = self.linear_1(x)
        x = self.linear_2(x)
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

    NUM_FEATS = 1024*2
    NUM_CLASSES = 1024
    DATASET_SIZE = 1000


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
    
    model.load(f"data/origin_model_{rank}.pkl")

    model.distributed(rpc = "gloo", rank = rank, world_size = world)

    q_model = quantize(model)
    # q_model.eval()

    peft_model = get_peft_model(q_model)
    
    fdsp_model = FSDPModule(
        peft_model,
        # use_orig_params = True,
        # sync_module_states = True,
        # auto_wrap_policy = my_auto_wrap_policy,
        # policy = ModuleWrapPolicy([nn.Linear,]),
        device_id=torch.device("cpu"),
    )

    # for p in fdsp_model.parameters():
    #     print(p, p.shape)

    optimizer = optim.Adam(fdsp_model.parameters(), lr = 1e-3)

    state_path = f"data/fsdp_model_{rank}.pkl"

    # fdsp_model.load(state_path)

    print('before fit:', fdsp_model(X[0]).sum())

    # inputs = torch.rand(10, NUM_FEATS)
    # fdsp_model.fit(loader, epoch = 20, early_stopping = False)
    train(fdsp_model, loader, epoch = 20)

    print('after fit:', fdsp_model(X[0]).sum())

    print(fdsp_model)
    print("##### fsdp parameters:", get_parameters(fdsp_model).shape)
    print("##### fsdp q model flatten:", fdsp_model.linear_2._handle.flat_param)
    print("##### q_model parameters:", type(get_parameters(q_model.linear_2)))

    # out = fdsp_model(inputs).sum()

    # out.backward()

    # print("~~~~~", out)

    # print(model)
    model.save(f"data/origin_model_{rank}.pkl")
    
    fdsp_model.save(state_path)



def train(model, loader, **kwargs):
    from ..trainer import Trainer
    trainer = Trainer(model, loader, early_stopping = False)


    @trainer.fit_step
    def fit_step(model, batch):
        x, y = batch
        y_hat = model(x)
        # return F.cross_entropy(y_hat, y)
        return F.mse_loss(y_hat, y)
        
    trainer.train(**kwargs)

def get_parameters(model):
    return next(model.parameters())


def quantize(model):
    import copy
    from quanto import Calibration, freeze, qfloat8, qint4, qint8, quantize

    m_copy = copy.deepcopy(model)

    quantize(m_copy, weights=qint4)
    freeze(m_copy)

    # m_copy = replace_hqq_linear(m_copy)

    # m_copy = replace_qlinear(m_copy)

    print("### q model linear_2 weight", m_copy.linear_2.weight)
    print("### q model linear_2 parameters", get_parameters(m_copy.linear_2).dtype)

    return m_copy



def replace_qlinear(model, skip_modules=["lm_head"], **kwargs):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """
    from ..quantize import QLinear

    for name, module in model.named_children():
        if name in skip_modules:
            continue

        if isinstance(module, torch.nn.Linear):
            model._modules[name] = QLinear.qcreate(module, **kwargs)
            model._modules[name].quantize()
            model._modules[name].freeze()
    
    return model



def replace_hqq_linear(model, skip_modules=["lm_head"], **kwargs):
    """
    Replace linear modules with a new Linear module.
    Parameters:
        model (`torch.nn.Module`):
            Input model or `torch.nn.Module` as the function is run recursively.
        skip_modules (`List[str]`, *optional*, defaults to `lm_head`):
            List of modules names not to convert. Defaults to `lm_head`.
    """
    from hqq.core.quantize import HQQLinear, HQQBackend, BaseQuantizeConfig
    
    quant_config = BaseQuantizeConfig(
        nbits=4,
        group_size=64,
        # quant_zero=True,
        # quant_scale=True,
        # offload_meta=True,
        view_as_float=True
    )

    for name, module in model.named_children():
        if name in skip_modules:
            continue
        
        if len(list(module.children())) > 0:
            replace_linear(module, HQQLinear, quant_config, skip_modules, **kwargs)

        if isinstance(module, torch.nn.Linear):
            model._modules[name] = HQQLinear(module, quant_config, **kwargs)
    
    HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
    return model



def get_peft_model(model):
    from peft import get_peft_config, get_peft_model, LoraConfig, TaskType

    peft_config = LoraConfig(
        # task_type=TaskType.SEQ_2_SEQ_LM,
        # task_type=TaskType.FEATURE_EXTRACTION,
        target_modules = ['linear_1'],
        inference_mode=False,
        r=8,
        lora_alpha=32,
        lora_dropout=0.1,
    )

    model = get_peft_model(model, peft_config)
    
    return model



def test_fsdp_model():
    import torch.multiprocessing as mp
    
    import os
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'

    mp.spawn(worker, args=(2,), nprocs=2, join=True)

