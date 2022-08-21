from time import sleep, time
from .progress import Progress


class TestIterator:
    def __init__(self, size):
        self._size = size
    
    def __iter__(self):
        for i in range(self._size):
            yield i


def test_progress():
    p =  Progress(range(100))
    for i in p:
        sleep(0.01)
    assert p.idx == 100

def test_progress_size():
    p = Progress(range(9527))
    assert p.size == 9527

def test_iterator():
    ti = TestIterator(100)
    p = Progress(ti)
    for i in p:
        sleep(0.01)
    assert p.idx == 100


def test_multi_loop():
    p = Progress(range(100))
    for i in p:
        sleep(0.01)
    assert p.idx == 100
    
    for i in p:
        sleep(0.01)
    assert p.idx == 100

def test_speed():
    p = Progress(range(1000))
    for i in p:
        sleep(0.001)
    assert p.idx == 1000
