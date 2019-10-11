import seaborn as sns
from .utils import (
    get_axes,
    tadpole_axes,
    FIG_SIZE,
)

class Tadpole:
    def __getattr__(self, name):
        t = getattr(sns, name)
        if callable(t):
            return self.wrapsns(t)

        return t

    def wrapsns(self, f):
        @tadpole_axes
        def wrapper(*args, figure_size = FIG_SIZE, **kwargs):
            kw = kwargs.copy()
            if 'ax' not in kw:
                kw['ax'] = get_axes(size = figure_size)

            try:
                return f(*args, **kw)
            except:
                return f(*args, **kwargs)

        return wrapper
