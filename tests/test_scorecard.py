import unittest
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


class TestScoreCard(unittest.TestCase):
    def setUp(self):
        pass

    def test_proba_to_score(self):
        proba = model.predict_proba(woe)[:,1]
        score = card.proba_to_score(proba)
        self.assertEqual(score[404], 456.66402014254516)

    def test_predict(self):
        score = card.predict(df)
        self.assertEqual(score[404], 456.66402014254516)

    def test_predict_sub_score(self):
        score, sub = card.predict(df, return_sub = True)
        self.assertEqual(sub.iloc[250, 1], 235.0048506817883)

    def test_woe_to_score(self):
        score = card.woe_to_score(woe)
        score = np.sum(score, axis = 1)
        self.assertEqual(score[404], 456.66402014254516)

    def test_bin_to_score(self):
        score = card.bin_to_score(bins)
        self.assertEqual(score[404], 456.66402014254516)

    def test_export_map(self):
        card_map = card.export_map()
        self.assertEqual(card_map['B']['D'], 232.18437983377214)

    def test_card_map(self):
        config = card.export_map()
        card_from_map = ScoreCard(card = config)
        score = card_from_map.predict(df)
        self.assertEqual(score[404], 456.66402014254516)

    def test_card_map_with_else(self):
        card_from_map = ScoreCard(card = card_config)
        score = card_from_map.predict(df)
        self.assertEqual(score[80], 800)

    def test_generate_testing_frame(self):
        card = ScoreCard(card = card_config)
        frame = card.testing_frame()
        self.assertEqual(frame.loc[4, 'B'], 'E')
