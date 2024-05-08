

def get_cluster(backend = "mp"):
    if backend is "mp":
        from .torch import TorchCluster
        return TorchCluster()
    else:
        return None
