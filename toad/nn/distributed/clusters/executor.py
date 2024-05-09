from typing import Callable
from dataclasses import dataclass
from ...trainer.trainer import TrainerState, Trainer
from ..accelerate.strategy import Strategy

@dataclass
class ExecutorContext:
    rank: int = -1
    size: int = 0
    trainer: Trainer = None
    func: Callable = None
    strategy: Strategy = None


class Executor:
    def __init__(self, context: ExecutorContext):
        self.context = context
    
    @property
    def rank(self):
        return self.context.rank

    def accelerator_prepare(self):
        from ..accelerate.accelerator import Accelerator
        
        accelerator = Accelerator(
            rank = self.context.rank,
            size = self.context.size,
            strategy = self.context.strategy,
        )

        print("~~~~~rank:", accelerator.rank)
        print("~~~~~size:", accelerator.size)

        module, loader, optimizer = accelerator.prepare(
            module = self.context.trainer.module,
            loader = self.context.trainer.loader,
            optimizer = self.context.trainer.optimizer,
        )

        return module, loader, optimizer, accelerator
    

    def prepare_trainer(self, trainer):
        from ..accelerate.accelerator import Accelerator
        
        accelerator = Accelerator(
            rank = self.context.rank,
            size = self.context.size,
            strategy = self.context.strategy,
        )

        module, loader, optimizer = accelerator.prepare(
            module = self.context.trainer.module,
            loader = self.context.trainer.loader,
            optimizer = self.context.trainer.optimizer,
        )

        trainer.state.module = module
        trainer.state.loader = loader
        trainer.state.optimizer = optimizer

        return trainer

    def run(self, *args, **kwargs):
        trainer = self.prepare_trainer(self.context.trainer)

        module = trainer.state.module
        loader = trainer.state.loader

        import torch
        torch.manual_seed(self.rank)

        self.context.func(self.context.trainer, loader, **kwargs)

        # TODO: save module
        import torch
        from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
        from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

        with FSDP.state_dict_type(module, StateDictType.SHARDED_STATE_DICT):
            torch.save(module.state_dict(), f"examples/model_{self.rank}.pt")
        
        return module
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
