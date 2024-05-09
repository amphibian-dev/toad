from dataclasses import dataclass

from .clusters.base import Cluster

@dataclass
class DistributedState:
    size: int = 0
    cluster: Cluster = None


class Distributor:
    def __init__(self, size, cluster = 'mp'):
        from .clusters import get_cluster

        self.state = DistributedState(
            size = size,
            cluster = get_cluster(cluster),
        )


    def init(self):
        pass

    def spawn(self, func, trainer, **kwargs):
        
        # HACK: remove trainer event
        from ..trainer.event import Event
        trainer.state.event = Event()

        self.state.cluster.spawn(func, self.state.size, trainer, **kwargs)
        
