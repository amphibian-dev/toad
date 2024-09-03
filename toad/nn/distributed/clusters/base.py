

class Cluster:
    def __init__(self):
        pass

    def spawn(self, func, size, **kwargs):
        # TODO: use python multiprocess
        import torch.multiprocessing as mp

        from .executor import Executor, ExecutorContext
        context = ExecutorContext(
            size = size,
            func = func,
            params = kwargs,
        )

        executor = Executor(context)

        mp.spawn(executor, nprocs = size, join = True)

