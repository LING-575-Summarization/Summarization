'''
Class to help with counting. Subclasses dict.
If a key is not in the dictionary, it adds it to zero
'''

from typing import *
from numbers import Number


class CounterDict(dict):
    '''Dict with a default value of 0. But, only assigns value when called'''
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

    def update(self, dictionary: dict):
        for key, value in dictionary.items():
            self[key] += value

    def update_from_wordset(self, wordset: Iterable):
        assert len(wordset) == len(set(wordset)), "Provided word set contains duplicate entries"
        for word in wordset:
            self[word] += 1