import pytest
import numpy as np
from .mixin import RulesMixin

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
    r = RulesObject().load(rules)
    assert r.rules['A']['rule'] == 'rule_A'

def test_rule_format():
    r = RulesObject().load(rules)
    assert r.export()['A'] == 'rule -> rule_A'

def test_save_update():
    r = RulesObject().load(rules)
    r.update({'A': 'update_A'})
    assert r.rules['A']['rule'] == 'update_A'
