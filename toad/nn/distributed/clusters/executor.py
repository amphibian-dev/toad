from typing import Callable
from dataclasses import dataclass

from ..accelerate.accelerator import Accelerator
from ..accelerate.strategy import Strategy
from ...trainer.trainer import TrainerState, Trainer

@dataclass
class ExecutorContext:
    rank: int = -1
    size: int = 0
    trainer: Trainer = None
    func: Callable = None
    strategy: Strategy = None
    accelerator: Accelerator = None


class Executor:
    def __init__(self, context: ExecutorContext):
        self.context = context
    
    @property
    def rank(self):
        return self.context.rank
    
    @property
    def accelerator(self):
        return self.context.accelerator
    

    def prepare_trainer(self, trainer):
        if self.accelerator is None:
            self.context.accelerator = Accelerator(
                rank = self.context.rank,
                size = self.context.size,
                strategy = self.context.strategy,
            )

        module, loader, optimizer = self.accelerator.prepare(
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

        self.context.func(self.context.trainer, **kwargs)

        self.accelerator.save(module)
        
        return module
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
