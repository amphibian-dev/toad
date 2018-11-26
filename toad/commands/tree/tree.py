"""
Windows:
    conda install python-graphviz
Mac:
    brew install graphviz
    pip install graphviz
"""

import numpy as np
import pandas as pd

import graphviz

import sklearn
from sklearn.tree import DecisionTreeClassifier


def tree_to_dot(tree, features, high_light = 0.15):
    from io import StringIO
    from sklearn.tree import _tree

    out = StringIO()
    tree_ = tree.tree_

    features = np.array([
        features[i] if i != _tree.TREE_UNDEFINED else "undefined!"
        for i in tree_.feature
    ])

    out.write('digraph Tree {\n')
    out.write('edge [fontname="FangSong"];\n')
    out.write('node [shape=box];\n')

    def recurse(node, parent = None, label = None):
        sample = tree_.n_node_samples[node]
        bad_rate = tree_.value[node][0,1] / sample

        out.write('{} [label="'.format(node))

        out.write('bad rate: {:.2%}\n'.format(bad_rate))
        out.write('sample: {:.2%}\n'.format(sample / tree_.n_node_samples[0]))

        # end of label
        out.write('"')

        if bad_rate > high_light:
            out.write(', color="red"')

        # end of node
        out.write('];\n')

        if tree_.feature[node] != _tree.TREE_UNDEFINED:
            name = features[node]
            threshold = tree_.threshold[node]
            recurse(tree_.children_left[node], node, '{} <= {:.2f}'.format(name, threshold))
            recurse(tree_.children_right[node], node, '{} > {:.2f}'.format(name, threshold))

        if parent is not None:
            out.write('{} -> {} [label="{}"];\n'.format(parent, node, label))

    recurse(0, None)

    out.write('}')
    s = out.getvalue()
    out.close()
    return s


def dot_to_img(dot, file = 'report.png'):
    import os

    name, ext = os.path.splitext(file)

    graph = graphviz.Source(dot)
    graph.format = ext[1:]
    graph.view(name, cleanup = True)


def split_data(frame, target = 'target'):
    X = frame.drop(columns = target)

    res = (X,)
    if isinstance(target, str):
        target = [target]

    for col in target:
        res += (frame[col],)

    return res


def dtree(frame, target, criterion = 'gini', depth = None, sample = 0.01, ratio = 0.15):
    tree = DecisionTreeClassifier(
        criterion = criterion,
        min_samples_leaf = sample,
        max_depth = depth,
    )

    tree.fit(frame.fillna(-1), target)

    dot_string = tree_to_dot(tree, frame.columns.values, high_light = ratio)

    dot_to_img(dot_string, file = target.name + '.png')
