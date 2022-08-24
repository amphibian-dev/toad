from .callback import callback as Callback


class Event:
    def __init__(self):
        self._events = {}

    def register(self, event, handler):
        """register event into trainer
        """
        if not isinstance(handler, Callback):
            handler = Callback(handler)
        
        if event not in self._events:
            self._events[event] = []
        
        self._events[event].append(handler)


    def on(self, event):
        def wrapper(handler):
            self.register(event, handler)
            return handler

        return wrapper
    

    def emit(self, event, *args, **kwargs):
        """emit event
        """
        if event not in self._events:
            return
        
        # trigger handler
        for handler in self._events[event]:
            handler(*args, **kwargs)
    

    def mute(self, event):
        """remove events handlers
        """
        if event in self._events:
            handlers = self._events.pop(event)
