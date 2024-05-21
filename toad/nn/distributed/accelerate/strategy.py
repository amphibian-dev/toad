from dataclasses import dataclass


@dataclass
class Strategy:
    method: str = None
    output_dir: str = None

    def save(self, module):
        pass


@dataclass
class DDPStrategy(Strategy):
    method: str = "ddp"


@dataclass
class FSDPStrategy(DDPStrategy):
    method: str = "fsdp"
    policy: str = None
    device: str = None

    def save(self, module, rank = -1):
        # TODO: save module
        import torch
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        with FSDP.state_dict_type(module, StateDictType.SHARDED_STATE_DICT):
            torch.save(module.state_dict(), f"{self.output_dir}/model_{rank}.pt")
        

    def init_fn(self, rank, device = None):
        import torch

        device = device or torch.device("cpu")
        
        def fn(module):
            module.to_empty(device = device, recurse = False)
        
        if rank != 0:
            return fn
        
        return None
