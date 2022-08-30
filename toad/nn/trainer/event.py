from .callback import callback as Callback


class Event:
    def __init__(self):
        self._events = {}

    def register(self, event, handler, every = 1):
        """register events handler
        """
        if not isinstance(handler, Callback):
            handler = Callback(handler)
        
        if event not in self._events:
            self._events[event] = []

        handler._event_count = 0
        handler._event_every = every
        
        self._events[event].append(handler)


    def on(self, event, **kwargs):
        def wrapper(handler):
            self.register(event, handler, **kwargs)
            return handler

        return wrapper
    

    def emit(self, event, *args, **kwargs):
        """emit event
        """
        if event not in self._events:
            return
        
        # trigger handler
        for handler in self._events[event]:
            # increase count
            handler._event_count += 1

            # trigger event
            if handler._event_count % handler._event_every == 0:
                handler(*args, **kwargs)
    

    def mute(self, event):
        """remove events handlers
        """
        if event in self._events:
            handlers = self._events.pop(event)
