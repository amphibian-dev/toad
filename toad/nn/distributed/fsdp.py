import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload,
    BackwardPrefetch,
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    enable_wrap,
    wrap,
)


class FSDPModule(FSDP):
    """FSDP module class
    """
    def __init__(self, module, policy = None, *args, **kwargs):
        import functools
        from torch.distributed.fsdp.wrap import (
            size_based_auto_wrap_policy,
            ModuleWrapPolicy,
            enable_wrap,
            wrap,
        )
        my_auto_wrap_policy = functools.partial(
            size_based_auto_wrap_policy, min_num_params=10,
        )

        super().__init__(
            module,
            auto_wrap_policy = my_auto_wrap_policy,
            # auto_wrap_policy = policy,
            *args, 
            **kwargs
        )


    def fit(self, loader, trainer = None, optimizer = None, loss = None, early_stopping = None, **kwargs):
        """train model

        Args:
            loader (DataLoader): loader for training model
            trainer (Trainer): trainer for training model
            optimizer (torch.Optimier): the default optimizer is `Adam(lr = 1e-3)`
            loss (Callable): could be called as 'loss(y_hat, y)'
            early_stopping (earlystopping): the default value is `loss_earlystopping`, 
                you can set it to `False` to disable early stopping
            epoch (int): number of epoch for training loop
            callback (callable): callable function will be called every epoch
        """
        if trainer is None:
            from ..trainer import Trainer
            trainer = Trainer(self, loader, optimizer = optimizer, loss = loss, early_stopping = early_stopping)
            trainer.fit_step(self.module.__class__.fit_step)
        
        trainer.train(**kwargs)
    
    def save(self, path):
        """save shards state dict
        """
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        with FSDP.state_dict_type(self, StateDictType.SHARDED_STATE_DICT):
            torch.save(self.state_dict(), path)
    
    def load(self, path, *args, **kwargs):
        """load shards state dict
        """
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
    
        with FSDP.state_dict_type(self, StateDictType.SHARDED_STATE_DICT):
            self.load_state_dict(torch.load(path))
        
        return self
    

    def log(self, *args, **kwargs):
        return self.module.log(*args, **kwargs)
