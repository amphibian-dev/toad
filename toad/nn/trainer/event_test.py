from .event import Event


def test_event_trigger():
    e = Event()

    counts = 0

    @e.on("test:trigger")
    def func():
        nonlocal counts
        counts += 1
    
    e.emit("test:trigger")

    assert counts == 1


def test_event_trigger_every():
    e = Event()

    counts = 0

    @e.on("test:trigger", every = 2)
    def func():
        nonlocal counts
        counts += 1
    
    for i in range(10):
        e.emit("test:trigger")

    assert counts == 5
