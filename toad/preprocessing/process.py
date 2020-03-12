from ..utils.func import flatten_columns


class Processing:
    def __init__(self, data):
        self.data = data
        self.workers = []

    def groupby(self, name):
        self.groupby = name
        return self
    
    def agg(self, worker):
        self.workers.append(worker)
        return self
    
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
            
            res = res.join(data)
        
        return res
            

    
    def process(self, data):
        group = data.groupby(self.groupby)

        res = None
        for func in self.workers:
            r = group.agg(func)

            if res is None:
                res = r
                continue
            
            res = res.join(r)

        res.columns = flatten_columns(res.columns)
        return res
    


class VAR:
    def __init__(self, column):
        self.column = column
        self.operators = []
    
    def push(self, op, value):
        self.operators.append({
            'op': '__'+ op +'__',
            'value': value,
        })

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