

def get_cluster(backend = "mp"):
    if backend == "mp":
        from .torch import TorchCluster
        return TorchCluster()
    elif backend == "base":
        from .base import Cluster
        return Cluster()
    else:
        return None
