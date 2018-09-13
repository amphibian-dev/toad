import numpy as np
from .stats import WOE

from .utils import to_ndarray, np_count

def woe_transform(feature, target):
    """transform feature by woe
    """

    feature = to_ndarray(feature)
    target = to_ndarray(target)

    t_counts_0 = np_count(target, 0, default = 1)
    t_counts_1 = np_count(target, 1, default = 1)

    f = np.zeros(len(feature))

    for v in np.unique(feature):
        mask =feature == v
        sub_target = target[mask]
        
        sub_0 = np_count(sub_target, 0, default = 1)
        sub_1 = np_count(sub_target, 1, default = 1)

        y_prob = sub_1 / t_counts_1
        n_prob = sub_0 / t_counts_0

        f[mask] = WOE(y_prob, n_prob)

    return f
