from dataclasses import dataclass


@dataclass
class Strategy:
    method: str = None
    output_dir: str = None

    def save(self, module):
        pass


    def prepare_module(self, module, rank):
        pass


    def prepare_optimizer(self, optimizer, module):
        opt_cls = type(optimizer)
        params = optimizer.param_groups[0]
        params.pop("params")

        return opt_cls(module.parameters(), **params)



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
    

    def prepare_module(self, module, rank):
        from ..fsdp import FSDP
        from torch.distributed.fsdp import CPUOffload

        module = FSDP(
            module,
            sync_module_states = True if self.device.type == 'cuda' else None,
            auto_wrap_policy = self.policy,
            device_id = self.device,
            param_init_fn = self.init_fn(rank = rank, device = self.device),
            cpu_offload = CPUOffload(offload_params = True) if self.device.type == 'cuda' else None,
            limit_all_gathers = True,
        )

        return module


