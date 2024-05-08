from typing import Callable
from dataclasses import dataclass
from ...trainer.trainer import TrainerState, Trainer

@dataclass
class ExecutorContext:
    rank: int = -1
    size: int = 0
    backend: str = "ddp"
    strategy: str = "ddp"
    trainer: Trainer = None
    func: Callable = None


class Executor:
    def __init__(self, context: ExecutorContext):
        self.context = context
    

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


    def run(self, *args, **kwargs):
        module, loader, optimizer, accelerator = self.accelerator_prepare()

        self.context.func(self.context.trainer, loader, **kwargs)
        
        return module
    
    def __call__(self, *args, **kwargs):
        return self.run(*args, **kwargs)
