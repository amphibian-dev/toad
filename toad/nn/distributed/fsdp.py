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
    """distributed module class
    """
    def fit(self, *args, **kwargs):
        return self.module.fit(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        return self.module.save(*args, **kwargs)
    
    def load(self, *args, **kwargs):
        return self.module.load(*args, **kwargs)
    
    def log(self, *args, **kwargs):
        return self.module.log(*args, **kwargs)
