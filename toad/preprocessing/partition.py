import numpy as np
import pandas as pd


class Partition:
    def partition(self, data):
        yield np.ones(len(data)).astype(bool), ''



class TimePartition(Partition):
    def __init__(self, base, filter, times):
        """partition data by time delta

        Args:
            base (str): column name of base time
            filter (str): column name of target time to be compared
            times (list): list of time delta`
        
        Example:
            TimePartition('apply_time', 'query_time', ['30d', '90d', 'all'])
        """
        self.base = base
        self.filter = filter
        self.times = times
    

    def partition(self, data):
        """partition data

        Args:
            data (DataFrame): dataframe
        
        Returns:
            iterator -> ndarray[bool]: mask of partition data
            iterator -> str: suffix string of current partition
        """
        base = pd.to_datetime(data[self.base])
        filter = pd.to_datetime(data[self.filter])

        for t in self.times:
            if t != 'all':
                delta = pd.Timedelta(t)
                mask = filter > (base - delta)
            else:
                mask = np.ones(len(filter)).astype(bool)
            
            yield mask, '_' + t


class ValuePartition(Partition):
    def __init__(self, column):
        self.column = column
    

    def partition(self, data):
        data = data[self.column]
        unique = data.unique()

        for u in unique:
            if pd.isna(u):
                mask = data.isna()
            else:
                mask = (data == u)
            
            yield mask, '_' + str(u)