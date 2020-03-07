import pytest
from toad.utils.mixin import SaveMixin

class SaveObject(SaveMixin):
    @property
    def rules(self):
        return self._rules


def test_save_load():
    obj = SaveObject()
    obj.load({'A': 3, 'B': 6})
    assert obj.rules['A'] == 3


def test_save_export():
    obj = SaveObject()
    obj.load({'A': 3, 'B': 6})
    assert obj.export()['B'] == 6


def test_save_update():
    obj = SaveObject()
    obj.load({'A': 3, 'B': 6})
    obj.update({'A': 4})
    assert obj.rules['A'] == 4 and obj.rules['B'] == 6
