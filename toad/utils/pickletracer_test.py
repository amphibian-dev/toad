import sys
import pytest
from .pickletracer import Tracer, get_current_tracer


def test_tracer_with_clause():
    assert get_current_tracer() is None
    with Tracer() as t:
        assert get_current_tracer() == t
    
    assert get_current_tracer() is None


@pytest.mark.skipif(sys.platform == "win32", reason="does not run on windows")
def test_trace_pyfunc():
    import pandas as pd
    import numpy as np
    from sklearn.linear_model import LinearRegression
    X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
    # y = 1 * x_0 + 2 * x_1 + 3
    y = np.dot(X, np.array([1, 2])) + 3
    reg = LinearRegression().fit(X, y)
    reg.score(X, y)

    def func(data):
        # data = dfunc(data)
        df = pd.DataFrame(data)
        return df

    class Model:
        def __init__(self, model, pref):
            self.model = model
            self.pref = pref
        
        def predict(self, data):
            data = self.pref(data)
            return self.model.predict(data)


    m = Model(reg, func)

    deps = Tracer().trace(m)

    assert set([m.__name__ for m in deps['pip']]) == set(['numpy', 'pandas', 'cloudpickle', 'sklearn'])


def test_default_cloudpickle():
    import pandas as pd
    
    def func(data):
        # data = dfunc(data)
        df = pd.DataFrame(data)
        return df
    
    deps = Tracer().trace(func)

    import io
    import cloudpickle
    
    dummy = io.BytesIO()
    # this should be correct after trace object
    # test for restore cloudpickle global dispatch table
    cloudpickle.dump(func, dummy)
