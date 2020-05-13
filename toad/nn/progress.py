
class Progress:
    def __init__(self, iterable):
        self.iterable = iterable

        self.batch = 1
        if hasattr(iterable, 'batch_size'):
            self.batch = iterable.getattr('batch_size')
        
        self.size = len(iterable)
        self.idx = 0


        self.BAR_LENGTH = 32
        self.template = "{percent:.0%} |{bar}| [{done}/{size}] {time}"


    def __len__(self):
        return self.size
    

    def __iter__(self):
        for item in self.iterable:
            yield item
        
            self.idx += 1
            self.flush()
        
        print()

    def flush(self):
        done = self.idx * self.batch

        percent = done / self.size

        bar = ('#' * int(percent * self.BAR_LENGTH)).ljust(self.BAR_LENGTH, '.')

        print('\r' + self.template.format(
            percent = percent,
            bar = bar,
            done = done,
            size = self.size,
            time = '1',
        ), end = '')



