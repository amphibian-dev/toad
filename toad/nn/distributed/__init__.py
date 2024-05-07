

def get_distributed_module(module, backend = None, **kwargs):
    from .ddp import DDPModule
    from .fsdp import FSDPModule
    
    if backend == 'fsdp':
        return FSDPModule(module, **kwargs)
    
    return DDPModule(module, **kwargs)



def prepare(module, backend = None, **kwargs):
    from ..module import ModuleMixin
    
    if backend == 'fsdp':
        from .fsdp import FSDP
        module = FSDP(module, **kwargs)
    
    from .ddp import DDP
    module = DDP(module, **kwargs)
    
    return ModuleMixin.mixin(module)
