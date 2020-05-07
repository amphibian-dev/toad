import pytest
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .scorecard import ScoreCard, WOETransformer, Combiner


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
        'nan': 500,
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



# create a score card
card = ScoreCard(
    combiner = combiner,
    transer = woe_transer,
)
card.fit(woe, target)

FUZZ_THRESHOLD = 1e-4
TEST_SCORE = pytest.approx(453.58, FUZZ_THRESHOLD)



def test_load():
    card = ScoreCard().load(card_config)
    score = card.predict(df)
    assert score[200] == 600


def test_proba_to_score():
    model = LogisticRegression()
    model.fit(woe, target)

    proba = model.predict_proba(woe)[:,1]
    score = card.proba_to_score(proba)
    assert score[404] == TEST_SCORE

def test_score_to_prob():
    score = card.predict(df)
    proba = card.score_to_proba(score)
    assert proba[404] == 0.4673929989138551

def test_predict():
    score = card.predict(df)
    assert score[404] == TEST_SCORE

def test_predict_proba():
    proba = card.predict_proba(df)
    assert proba[404,1] == 0.4673929989138551

def test_predict_sub_score():
    score, sub = card.predict(df, return_sub = True)
    assert sub.loc[250, 'B'] == pytest.approx(162.0781460573475, FUZZ_THRESHOLD)

def test_woe_to_score():
    score = card.woe_to_score(woe)
    score = np.sum(score, axis = 1)
    assert score[404] == TEST_SCORE

def test_bin_to_score():
    score = card.bin_to_score(bins)
    assert score[404] == TEST_SCORE

def test_export_map():
    card_map = card.export()
    assert card_map['B']['D'] == 159.24

def test_card_map():
    config = card.export()
    card_from_map = ScoreCard().load(config)
    score = card_from_map.predict(df)
    assert score[404] == TEST_SCORE

def test_card_map_with_else():
    card_from_map = ScoreCard().load(card_config)
    score = card_from_map.predict(df)
    assert score[80] == 1000

def test_generate_testing_frame():
    card = ScoreCard().load(card_config)
    frame = card.testing_frame()
    assert frame.loc[4, 'B'] == 'E'

def test_export_frame():
    card = ScoreCard().load(card_config)
    frame = card.export(to_frame = True)
    rows = frame[(frame['name'] == 'B') & (frame['value'] == 'else')].reset_index()
    assert rows.loc[0, 'score'] == 500

def test_card_combiner_number_not_match():
    c = combiner.export()
    c['A'] = [0, 3, 6, 8]
    com = Combiner().load(c)
    bins = com.transform(df)
    woe_transer = WOETransformer()
    woe = woe_transer.fit_transform(bins, target)

    card = ScoreCard(
        combiner = com,
        transer = woe_transer,
    )

    with pytest.raises(Exception) as e:
        # will raise an exception when fitting a card
        card.fit(woe, target)

    assert '\'A\' is not matched' in str(e.value)


def test_card_combiner_str_not_match():
    c = combiner.export()
    c['C'] = [['A'], ['B'], ['C']]
    com = Combiner().load(c)
    bins = com.transform(df)
    woe_transer = WOETransformer()
    woe = woe_transer.fit_transform(bins, target)

    card = ScoreCard(
        combiner = com,
        transer = woe_transer,
    )

    with pytest.raises(Exception) as e:
        # will raise an exception when fitting a card
        card.fit(woe, target)

    assert '\'C\' is not matched' in str(e.value)


def test_card_with_less_X():
    x = woe.drop(columns = 'A')
    card = ScoreCard(
        combiner = combiner,
        transer = woe_transer,
    )

    card.fit(x, target)
    assert card.predict(x)[200] == pytest.approx(411.968588097131, FUZZ_THRESHOLD)

