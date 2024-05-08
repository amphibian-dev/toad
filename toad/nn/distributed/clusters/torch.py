from .base import Cluster
from .executor import Executor, ExecutorContext


def _wrap(rank, size, func, *args, **kwargs):
    from toad.nn.distributed.accelerator import Accelerator
    accelerator = Accelerator(rank = rank, size = size)


class TorchExecutor(Executor):
    def __call__(self, rank, *args, **kwargs):
        self.context.rank = rank
        self.run(*args, **kwargs)


class TorchCluster(Cluster):
    def spawn(self, func, size, trainer):
        import torch.multiprocessing as mp

        context = ExecutorContext(
            trainer = trainer,
            size = size,
            func = func,
        )

        executor = TorchExecutor(context)

        mp.spawn(executor, nprocs = size, join = True)

