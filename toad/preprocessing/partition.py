import numpy as np
import pandas as pd


class Partition:
    def apply(self, data):
        return self



class TimePartition(Partition):
    def __init__(self, base, filter, times):
        self.base = base
        self.filter = filter
        self.times = times
    
    def apply(self, data):
        self.base = pd.to_datetime(data[self.base])
        self.filter = pd.to_datetime(data[self.filter])
        
        return self
    
    def __iter__(self):
        for t in self.times:
            if t != 'all':
                delta = pd.Timedelta(t)
                mask = self.filter > (self.base - delta)
            else:
                mask = np.ones(len(self.filter)).astype(bool)
            
            yield mask, '_' + t