from time import sleep
from .progress import Progress


def test_progress():
    for i in Progress(range(100)):
        sleep(0.01)
        pass


def test_progress_size():
    p = Progress(range(9527))
    assert p.size == 9527