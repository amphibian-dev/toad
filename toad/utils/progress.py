from time import time

class Progress:
    def __init__(self, iterable, size = None):
        self.iterable = iterable

        self.batch = 1
        self.size = size
        if hasattr(iterable, '__len__'):
            self.size = len(iterable)
        
        # is pytorch dataloader
        if hasattr(iterable, 'batch_size'):
            self.batch = getattr(iterable, 'batch_size')
            self.size = len(iterable.dataset)


        self.idx = 0
        self.time = None


        self.BAR_LENGTH = 32
        
        self.SYMBOL_DONE = '█'
        self.SYMBOL_REST = '.'
        self.prefix = ""
        self.suffix = ""
        self.template = "{prefix} {percent:3.0f}%|{bar}| [{done}/{size}] {time:.2f}s {suffix}"


    def __len__(self):
        return self.size
    

    def __iter__(self):
        start = time()
        for item in self.iterable:
            yield item

            self.time = time() - start
            self.idx += 1
            self.flush()
        
        # reset index
        self.idx = 0
        print()

    def flush(self):
        done = min(self.idx * self.batch, self.size)
        percent = done / self.size

        bar = (self.SYMBOL_DONE * int(percent * self.BAR_LENGTH)).ljust(self.BAR_LENGTH, self.SYMBOL_REST)

        print('\r' + self.template.format(
            percent = percent * 100,
            bar = bar,
            done = done,
            size = self.size,
            time = self.time,
            prefix = self.prefix,
            suffix = self.suffix,
        ), end = '')



