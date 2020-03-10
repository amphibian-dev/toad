import pytest
import numpy as np
from toad.utils.mixin import RulesMixin

np.random.seed(1)


class RulesObject(RulesMixin):
    def _parse_rule(self, rule):
        return {
            'rule': rule
        }
    

    def _format_rule(self, rule):
        return 'rule -> %s' % rule['rule']


rules = {'A': 'rule_A'}



def test_rule_parse():
    r = RulesObject()
    r.load(rules)
    assert r.rules['A']['rule'] == 'rule_A'


def test_rule_format():
    r = RulesObject()
    r.load(rules)
    assert r.export()['A'] == 'rule -> rule_A'
