import re
import io
import math
import json
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator

from .transform import WOETransformer, Combiner, ELSE_GROUP, EMPTY_BIN
from .utils import to_ndarray, bin_by_splits


RE_NUM = '-?\d+(.\d+)?'
RE_SEP = '[~-]'
RE_BEGIN = '(-inf|{num})'.format(num = RE_NUM)
RE_END = '(inf|{num})'.format(num = RE_NUM)
RE_RANGE = '\[{begin}\s*{sep}\s*{end}\)'.format(
    begin = RE_BEGIN,
    end = RE_END,
    sep = RE_SEP,
)


BORDER_THRESHOLD = 1e-6
NUMBER_EMPTY = -9999999
NUMBER_INF = 1e10
FACTOR_EMPTY = 'MISSING'
FACTOR_UNKNOWN = 'UNKNOWN'



class ScoreCard(BaseEstimator):
    def __init__(self, pdo = 60, rate = 2, base_odds = 35, base_score = 750,
                card = None, combiner = None, transer = None, model = None):
        """
        """
        self.pdo = pdo
        self.rate = rate
        self.base_odds = base_odds
        self.base_score = base_score

        self.factor = pdo / np.log(rate)
        self.offset = base_score - self.factor * np.log(base_odds)

        self.generate_card(
            card = card,
            combiner = combiner,
            transer = transer,
            model = model,
        )


    def generate_card(self, card = None, combiner = None, transer = None, model = None):
        """

        Args:
            card (dict|IOBase): dict of card or io to read json
            combiner (toad.Combiner)
            transer (toad.WOETransformer)
            model (LogisticRegression)
        """
        if card is not None:
            if isinstance(card, io.IOBase):
                with card as f:
                    card = json.load(f)

            return self.set_card(card)

        if combiner is not None:
            self.set_combiner(combiner)

        if transer is not None and model is not None:
            map = self.generate_map(transer, model)
            self.set_score(map)

        return self


    def fit(self, X, y):
        """
        Args:
            X (2D array-like)
            Y (array-like)

        """

        return self


    def _parse_range(self, bins):
        exp = re.compile(RE_RANGE)

        l = list()
        for item in bins:
            m = exp.match(item)

            # if is not range
            if m is None:
                return None

            # get the end number of range
            split = m.group(3)
            if split == 'inf':
                split = np.inf
            else:
                split = float(split)

            l.append(split)

        return np.array(l)


    def _parse_card(self, card):
        bins = card.keys()
        scores = card.values()
        scores = np.array(list(scores))

        groups = self._parse_range(bins)
        # if is continuous
        if groups is not None:
            ix = np.argsort(groups)
            scores = scores[ix]
            groups = groups[ix[:-1]]
        else:
            groups = list()
            for item in bins:
                if item == ELSE_GROUP:
                    groups.append(item)
                else:
                    groups.append(item.split(','))
            groups = np.array(groups)

        return groups, scores


    def set_card(self, card):
        """set card dict
        """
        combiner = dict()
        map = dict()
        for feature in card:
            bins, scores = self._parse_card(card[feature])
            combiner[feature] = bins
            map[feature] = scores

        self.set_combiner(combiner)
        self.set_score(map)

        return self


    def set_combiner(self, combiner):
        """set combiner
        """
        if not isinstance(combiner, Combiner):
            combiner = Combiner().set_rules(combiner)

        self.combiner = combiner


    def set_score(self, map):
        """set score map by dict
        """
        sm = dict()
        for key in map:
            s = map[key]
            if isinstance(s, np.ndarray):
                sm[key] = np.copy(s)
            else:
                sm[key] = np.array(s)

        self.score_map = sm


    def predict(self, X, **kwargs):
        """predict score
        Args:
            X (2D array-like): X to predict
            return_sub (bool): if need to return sub score, default `False`

        Returns:
            array-like: predicted score
            DataFrame: sub score for each feature
        """
        select = list(self.score_map.keys())
        bins = self.combine(X[select])
        return self.bin_to_score(bins, **kwargs)

    def proba_to_score(self, prob):
        """covert probability to score
        """
        odds = (1 - prob) / prob
        return self.factor * np.log(odds) + self.offset

    def combine(self, X):
        return self.combiner.transform(X)


    def bin_to_score(self, bins, return_sub = False):
        """predict score from bins
        """
        res = bins.copy()
        for col in self.score_map:
            s_map = self.score_map[col]
            b = bins[col].values
            # set default group to min score
            b[b == EMPTY_BIN] = np.argmin(s_map)
            # replace score
            res[col] = s_map[b]

        score = np.sum(res.values, axis = 1)

        if return_sub is False:
            return score

        return score, res

    def woe_to_score(self, woe, weight = None):
        """calculate score by woe
        """
        woe = to_ndarray(woe)

        if weight is None:
            weight = self.weight

        b = self.offset - self.factor * self.bias
        s = -self.factor * weight * woe

        # drop score whose weight is 0
        mask = 1
        if isinstance(weight, np.ndarray):
            mask = (weight != 0).astype(int)

        return (s + b / self.n_features_) * mask


    def set_model(self, model):
        """set logistic regression model
        """
        self.weight = model.coef_[0]
        self.bias = model.intercept_[0]
        self.n_features_ = (self.weight != 0).sum()


    def generate_map(self, transer, model):
        """calculate score map by woe
        """
        self.set_model(model)

        keys = list(transer.values_.keys())

        s_map = dict()
        for i, k in enumerate(keys):
            weight = self.weight[i]
            # skip feature whose weight is 0
            if weight == 0:
                continue

            woe = transer.woe_[k]
            s_map[k] = self.woe_to_score(woe, weight = weight)

        return s_map


    def export(self, to_frame = False, to_json = None, to_csv = None):
        """generate a scorecard object

        Args:
            to_frame (bool): return DataFrame of card
            to_json (IOBase): io to write json file
            to_csv (filepath|IOBase): file to write csv

        Returns:
            dict
        """
        card = dict()
        combiner = self.combiner.export(format = True)

        for col in self.score_map:
            bins = combiner[col]
            card[col] = dict()

            for i in range(len(bins)):
                card[col][bins[i]] = self.score_map[col][i]

        if to_json is not None:
            with to_json as f:
                return json.dump(card, f, ensure_ascii = False)

        if to_frame or to_csv is not None:
            rows = list()
            for name in card:
                for value, score in card[name].items():
                    rows.append({
                        'name': name,
                        'value': value,
                        'score': score,
                    })

            card = pd.DataFrame(rows)


        if to_csv is not None:
            return card.to_csv(to_csv)

        return card



    def _generate_testing_frame(self, size = 'max', mishap = True):
        """
        Args:
            size (int|str): size of frame. 'max' (default), 'lcm'
            mishap (bool): is need to add mishap patch to test frame

        Returns:
            DataFrame
        """
        c_map = self.combiner.export()

        number_patch = np.array([NUMBER_EMPTY, NUMBER_INF])
        factor_patch = np.array([FACTOR_EMPTY, FACTOR_UNKNOWN])

        values = []
        cols = []
        for k, v in c_map.items():
            v = np.array(v)
            if np.issubdtype(v.dtype, np.number):
                items = np.concatenate((v, v - BORDER_THRESHOLD))
                patch = number_patch
            else:
                # remove else group
                mask = np.argwhere(v == ELSE_GROUP)
                if mask.size > 0:
                    v = np.delete(v, mask)

                items = np.concatenate(v)
                patch = factor_patch

            if mishap:
                # add patch to items
                items = np.concatenate((items, patch))

            cols.append(k)
            values.append(np.unique(items))

        # calculate length of values in each columns
        lens = [len(x) for x in values]

        # get size
        if isinstance(size, str):
            if size == 'lcm':
                size = np.lcm.reduce(lens)
            else:
                size = np.max(lens)

        stacks = dict()
        for i in range(len(cols)):
            l = lens[i]
            # generate indexes of value in column
            ix = list(np.arange(l)) * math.ceil(size / l)
            stacks[cols[i]] = values[i][ix[:size]]

        return pd.DataFrame(stacks)

    def testing_frame(self, **kwargs):
        """get testing frame with score

        Returns:
            DataFrame: testing frame with score
        """
        frame = self._generate_testing_frame(**kwargs)
        frame['score'] = self.predict(frame)

        return frame
