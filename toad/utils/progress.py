import sys
from time import time

class Progress:
    """
    """
    def __init__(self, iterable, size = None, interval = 0.1):
        """
        Args:
            iterable
            size (int): max size of iterable
            interval (float): update bar interval second, default is `0.1`
        
        Attrs:
            BAR_LENGTH (int): bar length, default is `32`
            SYMBOL_DONE (str): symbol indicating complation
            SYMBOL_REST (str): symbol indicating remaining
            prefix (str): string template before progress bar
            suffix (str): string template after progress bar
            template (str): string template for rendering, `{prefix} {bar} {suffix}`
        """
        self.iterable = iterable
        self.interval = interval

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

        if self.size is None:
            self.template = "{prefix} {done} iters {time:.2f}s {suffix}"
        else:
            self.template = "{prefix} {percent:3.0f}%|{bar}| [{done}/{size}] {time:.2f}s {suffix}"


    def __len__(self):
        return self.size
    

    def __iter__(self):
        self.reset()
        # reset time
        start = time()
        last_time = start
        for item in self.iterable:
            yield item

            self.idx += 1

            curr_time = time()
            self.time = curr_time - start

            # skip update if delta is too small
            if curr_time - last_time < self.interval:
                continue
            
            last_time = curr_time
            
            # update bar
            self.flush()
        
        # finally updating for the status of end
        self.flush()
        self.end()
    

    def reset(self):
        # reset index
        self.idx = 0
    

    def end(self):
        self.print('\n')


    def flush(self):
        if self.size is None:
            done = self.idx * self.batch
            percent = 0
            bar = None
        else:
            done = min(self.idx * self.batch, self.size)
            percent = done / self.size

            bar = (self.SYMBOL_DONE * int(percent * self.BAR_LENGTH)).ljust(self.BAR_LENGTH, self.SYMBOL_REST)

        self.print('\r' + self.template.format(
            percent = percent * 100,
            bar = bar,
            done = done,
            size = self.size,
            time = self.time,
            tps = done / self.time,
            prefix = self.prefix,
            suffix = self.suffix,
        ))
    

    def print(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()



