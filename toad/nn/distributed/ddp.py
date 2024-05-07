from torch.nn.parallel import DistributedDataParallel as DDP


class DDPModule(DDP):
    """distributed module class
    """
    def fit(self, *args, **kwargs):
        return self.module.fit(*args, **kwargs)
    
    def save(self, *args, **kwargs):
        return self.module.save(*args, **kwargs)
    
    def load(self, *args, **kwargs):
        return self.module.load(*args, **kwargs)
    
    def log(self, *args, **kwargs):
        return self.module.log(*args, **kwargs)
