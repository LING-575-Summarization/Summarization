'''
Class to help with counting. Subclasses dict.
If a key is not in the dictionary, it adds it to zero
'''

from typing import *
from numbers import Number
from collections import OrderedDict
import numpy as np

class CounterDict(OrderedDict):

    def __init__(self, keys: Optional[List] = None, **kwargs):
        if keys is None:
            super().__init__(**kwargs)
        else:
            __keys = {k:0 for k in keys}
            super().__init__(**__keys)

    def __setitem__(self, __key: str, __val: Number) -> None:
        assert isinstance(__val, Number), f"value {__val} is not int, float or complex"
        return super().__setitem__(__key, __val)

    def __getitem__(self, __key: str) -> int:
        if __key in self:
            return super().__getitem__(__key)
        else:
            return 0

    def map(self, func: Callable[[int], int]):
        for key, val in self.items():
            self[key] = func(val)
        return self
    
    def __mul__(self, dictionary: dict) -> dict:
        assert isinstance(dictionary, dict), 'CounterDict must be multiplied another dict'
        assert set(self.keys()) == set(dictionary.keys()), 'Dicts must have the same keys'
        for k in self.keys():
            self[k] *= dictionary[k]
        return self
    
    def to_numpy(self):
        return np.array(list(self.values()))
    
    def update(self, dictionary: dict):
        for key, value in dictionary.items():
            self[key] += value

    def update_from_wordset(self, wordset: Iterable):
        assert len(wordset) == len(set(wordset)), "Provided word set contains duplicate entries"
        for word in wordset:
            self[word] += 1