from ..utils.func import flatten_columns


class Processing:
    def __init__(self, data):
        self.data = data
        self.funcs = {}

    def groupby(self, name):
        self.groupby = name
        return self
    
    def apply(self, f):
        if not isinstance(f, dict):
            f = {
                '_all_': f
            }
        
        for k, v in f.items():
            self.append_func(k, v)
        
        return self
    
    def append_func(self, col, func):
        if not isinstance(func, (list, tuple)):
            func = [func]
        
        func = list(func)
        
        if col not in self.funcs:
            self.funcs[col] = []
        
        self.funcs[col] = self.funcs[col] + func
        
    
    def splitby(self, s):
        self.splits = s
        return self
    
    def exec(self):
        res = None

        for mask, suffix in self.splits.apply(self.data):
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
            g = group
            
            for f in l:
                name = f
                if callable(f):
                    name = f.__name__
                    r = g.apply(self._wrapper(col, f))
                else:
                    r = getattr(g[col], f)()
                
                if isinstance(r, pd.Series):
                    r = pd.DataFrame(r)

                res.append(r.add_prefix(col + '_'))
        
        return pd.concat(res, axis=1)
    

    def _wrapper(self, col, f):
        def func(data):
            if col != '_all_':
                data = data[col]
            
            r = f(data)

            if not isinstance(r, dict):
                r = {
                    f.__name__: r
                }

            return pd.Series(r)
        
        return func
    


class VAR:
    def __init__(self, column):
        self.column = column
        self.operators = []
    
    def push(self, op, value):
        self.operators.append({
            'op': '__'+ op +'__',
            'value': value,
        })
    
    def replay(self, data):
        base = data[self.column]
        for item in self.operators:
            v = item['value']

            if isinstance(v, VAR):
                v = v.replay(data)
            
            f = getattr(base, item['op'])

            if v is None:
                base = f()
                continue

            base = f(v)
        
        return base


    def __eq__(self, other):
        self.push('eq', other)
        return self
    
    def __lt__(self, other):
        self.push('lt', other)
        return self
    
    def __gt__(self, other):
        self.push('gt', other)
        return self
    
    def __le__(self, other):
        self.push('le', other)
        return self
    
    def __ge__(self, other):
        self.push('ge', other)
        return self
    
    def __invert__(self):
        self.push('invert', None)
        return self
    
    def __and__(self, other):
        self.push('and', other)
        return self
    
    def __or__(self, other):
        self.push('or', other)
        return self
    
    def __xor__(self, other):
        self.push('xor', other)
        return self