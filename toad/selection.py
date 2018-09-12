import numpy as np
from .stats import IV


def drop_empty(frame, threshold = 0.9, nan = None):
    """drop columns by empty
    """
    if nan is not None:
        df = frame.replace(nan, np.nan)
    else:
        df = frame.copy()

    if threshold < 1:
        threshold = len(df) * threshold

    drop_list = []
    for col in df:
        n = df[col].isnull().sum()
        if n > threshold:
            drop_list.append(col)

    return frame.drop(columns = drop_list)

def drop_corr(frame, target = None, threshold = 0.7, by = 'IV'):
    """drop columns by corr
    """
    t = target
    f = frame

    if target is not None:
        t = frame[target]
        f = frame.drop(columns = target)

    corr = f.corr()

    # get the graph of relationship
    ix, cn = np.where(np.triu(corr.values, 1) > threshold)
    graph = np.hstack([ix.reshape((-1, 1)), cn.reshape((-1, 1))])

    uni, counts = np.unique(graph, return_counts = True)

    # calc weights for nodes
    weights = np.zeros(len(corr.index))

    if by is 'IV':
        for ix in uni:
            weights[ix] = IV(frame[corr.index[ix]], target = t)

    drops = []
    while(True):
        # TODO deal with circle

        # get nodes with the most relationship
        nodes = uni[np.argwhere(counts == np.amax(counts))].flatten()

        # get node who has the min weights
        n = nodes[np.argsort(weights[nodes])[0]]

        # get nodes of 1 degree relationship of n
        i, c = np.where(graph == n)
        pairs = graph[(i, 1-c)]

        # if sum of 1 degree nodes greater than n
        # then delete n self
        # else delete all 1 degree nodes
        if weights[pairs].sum() > weights[n]:
            dro = [n]
        else:
            dro = pairs.tolist()

        # add nodes to drops list
        drops += dro

        # delete nodes from graph
        di, _ = np.where(np.isin(graph, dro))
        graph = np.delete(graph, di, axis = 0)

        # if graph is empty
        if len(graph) <= 0:
            break

        # update nodes and counts
        uni, counts = np.unique(graph, return_counts = True)


    return frame.drop(columns = corr.index[drops])
