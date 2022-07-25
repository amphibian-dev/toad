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


        self.reset()


        self.BAR_LENGTH = 32
        
        self.SYMBOL_DONE = '█'
        self.SYMBOL_REST = '.'
        self.prefix = ""
        self.suffix = ""

        if self.size is None:
            self.template = "{prefix} {done} iters {time:.2f}s {tps}it/s {suffix}"
        else:
            self.template = "{prefix} {percent:3.0f}%|{bar}| [{done}/{size}] {time:.2f}s {suffix}"


    def __len__(self):
        return self.size
    

    def __iter__(self):
        self.reset()
        self.iterator = iter(self.iterable)
        return self
    

    def __next__(self):
        try:
            return self.next()
        except StopIteration as e:
            self.end()
            raise e      
    

    def reset(self):
        # reset index
        self.idx = 0

        # reset time
        self.time = None
        self.start_time = time()
        self._last_time = self.start_time
        self.iterator = iter(self.iterable)
    

    def next(self):
        item = next(self.iterator)
        self.update()
        return item
    

    def update(self, idx = None, force = False):
        # update idx
        if idx is None:
            idx = self.idx + 1
        
        self.idx = idx

        curr_time = time()
        self.time = curr_time - self.start_time

        # skip update if delta is too small
        if not force and curr_time - self._last_time < self.interval:
            return
        
        self._last_time = curr_time
        
        # update bar
        self.flush()
    

    def end(self):
        """progress end
        """
        self.update(idx = self.idx, force = True)
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
            tps = done / max(self.time, 1),
            prefix = self.prefix,
            suffix = self.suffix,
        ))
    

    def print(self, text):
        sys.stdout.write(text)
        sys.stdout.flush()
    





