import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from toad.transform import WOETransformer, Combiner
from toad.scorecard import ScoreCard

np.random.seed(1)

ab = np.array(list('ABCDEFG'))
feature = np.random.randint(10, size = 500)
target = np.random.randint(2, size = 500)
str_feat = ab[np.random.choice(7, 500)]

df = pd.DataFrame({
    'A': feature,
    'B': str_feat,
    'C': ab[np.random.choice(2, 500)],
    'D': np.ones(500),
})

card_config = {
    'A': {
        '[-inf ~ 3)': 100,
        '[3 ~ 5)': 200,
        '[5 ~ 8)': 300,
        '[8 ~ inf)': 400,
    },
    'B': {
        ','.join(list('ABCD')): 200,
        ','.join(list('EF')): 400,
        'else': 500,
    },
    'C': {
        'A': 200,
        'B': 100,
    },
}

combiner = Combiner()
bins = combiner.fit_transform(df, target, n_bins = 5)
woe_transer = WOETransformer()
woe = woe_transer.fit_transform(bins, target)

model = LogisticRegression()
# fit model by woe
model.fit(woe, target)

# create a score card
card = ScoreCard(
    combiner = combiner,
    transer = woe_transer,
    model = model,
)

TEST_SCORE = 453.58201689869964



def test_proba_to_score():
    proba = model.predict_proba(woe)[:,1]
    score = card.proba_to_score(proba)
    assert score[404] == 453.5820168986997

def test_predict():
    score = card.predict(df)
    assert score[404] == TEST_SCORE

def test_predict_sub_score():
    score, sub = card.predict(df, return_sub = True)
    assert sub.iloc[250, 1] == 162.08878336572937

def test_woe_to_score():
    score = card.woe_to_score(woe)
    score = np.sum(score, axis = 1)
    assert score[404] == TEST_SCORE

def test_bin_to_score():
    score = card.bin_to_score(bins)
    assert score[404] == TEST_SCORE

def test_export_map():
    card_map = card.export()
    assert card_map['B']['D'] == 159.2498541513114

def test_card_map():
    config = card.export()
    card_from_map = ScoreCard(card = config)
    score = card_from_map.predict(df)
    assert score[404] == TEST_SCORE

def test_card_map_with_else():
    card_from_map = ScoreCard(card = card_config)
    score = card_from_map.predict(df)
    assert score[80] == 1000

def test_generate_testing_frame():
    card = ScoreCard(card = card_config)
    frame = card.testing_frame()
    assert frame.loc[4, 'B'] == 'E'

def test_export_frame():
    card = ScoreCard(card = card_config)
    frame = card.export(to_frame = True)
    assert frame.loc[6, 'value'] == 'else'

def test_card_without_combiner():
    transer = WOETransformer()
    woe_X = transer.fit_transform(df, target)

    model = LogisticRegression()
    model.fit(woe_X, target)

    card = ScoreCard(transer = transer, model = model)
    score, sub = card.predict(df, return_sub = True)

    assert score[404] == 460.9789823549386
