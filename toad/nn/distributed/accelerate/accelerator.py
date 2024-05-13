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
    def __init__(self, rank = None, size = None, strategy = None):
        self.state = AcceleratorState(
            rank = rank,
            size = size,
            strategy = strategy or DDPStrategy(),
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
    
    @property
    def device(self):
        import torch
        if self.strategy.device.type == 'cpu':
            return torch.device('cpu')
        
        return torch.device(f"cuda:{self.rank}")

    def setup(self):
        import torch
        
        if not self.initialized:
            # choose a rpc type
            rpc = 'nccl' if torch.distributed.is_nccl_available() else 'gloo'
            
            master_url = 'tcp://localhost:12355'

            torch.distributed.init_process_group(
                rpc,
                rank = self.rank,
                world_size = self.size,
                init_method = master_url,
            )
        
        if self.device.type == 'cuda':
            torch.cuda.set_device(self.device)
    

    def prepare(self, module, loader, optimizer):
        self.setup()

        module = self.prepare_module(module)
        optimizer = self.prepare_optimizer(optimizer, module)

        return module, loader, optimizer
    

    def prepare_module(self, module):
        from ...module import ModuleMixin

        if isinstance(self.strategy, FSDPStrategy):
            from ..fsdp import FSDP
            from torch.distributed.fsdp import CPUOffload

            module = FSDP(
                module,
                auto_wrap_policy = self.strategy.policy,
                device_id = self.device,
                cpu_offload = CPUOffload(offload_params = True) if self.device.type == 'cuda' else None,
            )

        elif isinstance(self.strategy, DDPStrategy):
            from ..ddp import DDP
            module = DDP(module)

        return module
        # return ModuleMixin.mixin(module)
    
    def prepare_optimizer(self, optimizer, module):
        opt_cls = type(optimizer)
        params = optimizer.param_groups[0]
        params.pop("params")

        return opt_cls(module.parameters(), **params)
    

    def save(self, module):
        if self.strategy.output_dir is None:
            return
        
        self.strategy.save(module, rank = self.rank)

