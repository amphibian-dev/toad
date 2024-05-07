from dataclasses import dataclass
from .strategy import Strategy, DDPStrategy, FSDPStrategy


@dataclass
class AcceleratorState:
    rank: int = -1
    size: int = 0
    strategy: Strategy = None
    
    @property
    def initialized(self):
        import torch

        return torch.distributed.is_initialized()


class Accelerator:
    def __init__(self, rank = None, size = None, strategy = "ddp"):
        self.state = AcceleratorState(
            rank = rank,
            size = size,
            strategy = strategy,
        )
    

    @property
    def rank(self):
        return self.state.rank
    
    @property
    def size(self):
        return self.state.size
    
    @property
    def initialized(self):
        return self.state.initialized
    
    @property
    def strategy(self):
        return self.state.strategy

    def setup(self):
        import torch
        
        if not self.initialized:
            # choose a rpc type
            rpc = 'nccl' if torch.distributed.is_nccl_available() else 'gloo'

            torch.distributed.init_process_group(
                rpc,
                rank = self.rank,
                world_size = self.size,
            )
    

    def prepare(self, module, loader, optimizer):
        self.setup()

        module = self.prepare_module(module)

        return module, loader, optimizer
    

    def prepare_model(self, module):
        from ...module import ModuleMixin

        if isinstance(self.strategy, FSDPStrategy):
            from ..fsdp import FSDP
            module = FSDP(module, **kwargs)

        if isinstance(self.strategy, DDPStrategy):
            from ..ddp import DDP
            module = DDP(module, **kwargs)
        
        return ModuleMixin.mixin(module)

