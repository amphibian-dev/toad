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

combiner = Combiner()
bins = combiner.fit_transform(df, target, n_bins = 5)
woe_transer = WOETransformer()
woe = woe_transer.fit_transform(bins, target)

model = LogisticRegression()
# fit model by woe
model.fit(woe, target)

# create a score card
card = ScoreCard()

# fit score card
card.fit(df, target,
    model = model,
    combiner = combiner,
)


class TestScoreCard(unittest.TestCase):
    def setUp(self):
        pass

    def test_proba_to_score(self):
        proba = model.predict_proba(woe)[:,1]
        score = card.proba_to_score(proba)
        self.assertEqual(score[404], 456.66402014254516)

    def test_predict(self):
        score= card.predict(df)
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
