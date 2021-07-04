from .callback import callback

def test_callback():
    @callback
    def hook(history, trainer):
        return history['a']
    
    res = hook(epoch = 1, trainer = None, history = {"a": 3})

    assert res == 3
