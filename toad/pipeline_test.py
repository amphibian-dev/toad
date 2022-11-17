import os
import pytest
import pandas as pd
from os.path import join
from toad.pipeline import Toad_Pipeline
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import load_iris

li = load_iris()
X = pd.DataFrame(li['data'], columns=li['feature_names'])
Y = li['target']
Xtrain, Xtest, Ytrain, Ytest = train_test_split(X, Y, test_size=0.3, random_state=42)

def test_pipeline():
    toad_pipe = Toad_Pipeline()

    toad_pipe = toad_pipe.fit(Xtrain, Ytrain)
    Xtrain_ = toad_pipe.transform(Xtrain)
    assert Xtrain_.loc[106, 'sepal length (cm)'] == pytest.approx(-1.2755429968271879)

def test_pipeline_with_update():
    toad_pipe = Toad_Pipeline()

    update_dict = {
        'sepal width (cm)' : [2.3, 2.8, 3.3],
    }

    toad_params = {
        'combiner__update_rules' : update_dict,
        'stepwise__skip' : True,
    }

    toad_pipe = toad_pipe.set_params(**toad_params)
    toad_pipe = toad_pipe.fit(Xtrain, Ytrain)
    bins = toad_pipe.combiner.export()   
    assert bins['sepal width (cm)'] == update_dict['sepal width (cm)']

def test_pipeline_grid_search():
    toad_pipe = Toad_Pipeline()

    gs_params = {
        'select__iv' : [0.02, 0.05, 0.1],
        'select__corr' : [0.5, 0.7],
        'combiner__method' : ['chi', 'dt'],
        'combiner__min_samples' : [0.05, 0.1],
    }

    toad_pipe.steps.append(('model', LogisticRegression()))

    gs = GridSearchCV(
        toad_pipe, 
        gs_params, 
        cv=KFold(5, random_state=42, shuffle=True), 
    )
    gs = gs.fit(Xtrain, Ytrain)
    assert gs.best_score_ == pytest.approx(0.8857142857142858)
    assert gs.best_params_ == {
        'combiner__method': 'chi', 
        'combiner__min_samples': 0.05, 
        'select__corr': 0.5, 
        'select__iv': 0.02
    }
    assert gs.score(Xtest, Ytest) == pytest.approx(0.9777777777777777)
    assert list(gs.best_estimator_[-1].coef_.ravel()) == [-2.2499017912050854, 2.3171634795681637, -0.06726168836306577]
    assert gs.best_estimator_.combiner.export() == {'sepal width (cm)': [2.5, 2.8, 2.9, 3.0, 3.1, 3.4], 'petal length (cm)': [3.0, 4.5, 4.8, 5.2]}


