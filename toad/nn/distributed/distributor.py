from dataclasses import dataclass


@dataclass
class DistributedState:
    size: int = 0
    backend: str = 'ddp'


class Distributor:
    def __init__(self, size, backend = 'ddp', cluster = 'mp'):
        self.state = DistributedState(
            size = size,
            backend = backend,
        )


    def init(self):
        pass

    def prepare(self, module, loader, optimizer, scheduler):
        pass


    def spawn(self, func, *args):
        
        pass
