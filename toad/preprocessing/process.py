import pandas as pd


_ALL_SYMBOL_ = '__all_symbol__'

class Processing:
    """

    Example:

    >>> (Processing(data)
    ...     .groupby('id')
    ...     .partitionby(TimePartition(
    ...         'base_time',
    ...         'filter_time',
    ...         ['30d', '60d', '180d', '365d', 'all']
    ...     ))
    ...     .apply({'A': ['max', 'min', 'mean']})
    ...     .apply({'B': ['max', 'min', 'mean']})
    ...     .apply({'C': 'nunique'})
    ...     .apply({'D': {
    ...         'f': len,
    ...         'name': 'normal_count',
    ...         'mask':  Mask('D').isin(['normal']),
    ...     }})
    ...     .apply({'id': 'count'})
    ...     .exec()
    ... )
    """
    def __init__(self, data):
        self.data = data
        self.funcs = {}
        self.partitions = None

    def groupby(self, name):
        """group data by name

        Args:
            name (str): column name in data
        """
        self.groupby = name
        return self
    
    def apply(self, f):
        """apply functions to data

        Args:
            f (dict|function): a config dict that keys are the column names and 
                values are the functions, it will take the column series as the
                functions argument. if `f` is a function, it will take the whole
                dataframe as the argument.
            
        """
        if not isinstance(f, dict):
            f = {
                _ALL_SYMBOL_: f
            }
        
        for k, v in f.items():
            self.append_func(k, v)
        
        return self
    

    def append_func(self, col, func):
        if not isinstance(func, (list, tuple)):
            func = [func]
        
        if col not in self.funcs:
            self.funcs[col] = []
        
        for f in func:
            self.funcs[col].append(self._convert_func(f))
    

    def _convert_func(self, f):
        if isinstance(f, F):
            return f
        
        if not isinstance(f, dict):
            f = {'f': f}
        
        return F(**f)
        
    
    def partitionby(self, p):
        """partition data to multiple pieces, processing will process to all the pieces

        Args:
            p (Partition)
        """
        self.partitions = p
        return self
    
    def exec(self):
        if self.partitions is None:
            return self.process(self.data)

        res = None
        for mask, suffix in self.partitions.partition(self.data):
            data = self.process(self.data[mask])
            data = data.add_suffix(suffix)

            if res is None:
                res = data
                continue
            
            res = res.join(data, how = 'outer')
        
        return res
            

    
    def process(self, data):
        group = data.groupby(self.groupby)

        res = []
        for col, l in self.funcs.items():
            for f in l:
                g = group

                if f.need_filter:
                    g = f.filter(data).groupby(self.groupby)
                
                if f.is_buildin:
                    r = getattr(g[col], f.name)()
                    r.name = f.name
                else:
                    if col == _ALL_SYMBOL_:
                        col = None
                    
                    r = g.apply(f, col = col)
                
                if isinstance(r, pd.Series):
                    r = pd.DataFrame(r)

                res.append(r.add_prefix(col + '_'))
        
        return pd.concat(res, axis=1)
    


class Mask:
    """a placeholder to select dataframe
    """
    def __init__(self, column = None):
        self.column = column
        self.operators = []
    
    def push(self, op, value):
        self.operators.append({
            'op': op,
            'value': value,
        })
    
    def replay(self, data):
        base = data
        if self.column is not None:
            base = data[self.column]

        for item in self.operators:
            v = item['value']

            if isinstance(v, Mask):
                v = v.replay(data)
            
            f = getattr(base, item['op'])

            if v is None:
                base = f()
                continue

            base = f(v)
        
        return base

    def __eq__(self, other):
        self.push('__eq__', other)
        return self
    
    def __lt__(self, other):
        self.push('__lt__', other)
        return self
    
    def __gt__(self, other):
        self.push('__gt__', other)
        return self
    
    def __le__(self, other):
        self.push('__le__', other)
        return self
    
    def __ge__(self, other):
        self.push('__ge__', other)
        return self
    
    def __invert__(self):
        self.push('__invert__', None)
        return self
    
    def __and__(self, other):
        self.push('__and__', other)
        return self
    
    def __or__(self, other):
        self.push('__or__', other)
        return self
    
    def __xor__(self, other):
        self.push('__xor__', other)
        return self
    
    def isin(self, other):
        self.push('isin', other)
        return self
    
    def isna(self):
        self.push('isna', None)
        return self



class F:
    """function class for processing
    """
    def __init__(self, f, name = None, mask = None):
        self.f = f

        if name is None:
            if self.is_buildin:
                name = f
            else:
                name = f.__name__
        
        self.__name__ = name

        self.mask = mask
    
    @property
    def name(self):
        return self.__name__

    @property
    def is_buildin(self):
        return isinstance(self.f, str)
    
    @property
    def need_filter(self):
        return self.mask is not None
    
    def __call__(self, data, *args, col = None, **kwargs):
        if col in data:
            data = data[col]

        r = self.f(data, *args, **kwargs)

        if not isinstance(r, dict):
            r = {
                self.name: r
            }

        return pd.Series(r)
    

    def filter(self, data):
        if self.mask is None:
            return data
        
        mask = self.mask
        if isinstance(self.mask, Mask):
            mask = self.mask.replay(data)
        
        return data[mask]

