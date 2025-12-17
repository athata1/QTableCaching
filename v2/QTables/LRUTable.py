import collections
from typing import Hashable, Tuple, Dict
from QTables.AbstractTable import AbstractQTable

class LRUQTable(AbstractQTable):
    def __init__(self, capacity: int, default_value: float = 0.0):
        super().__init__(default_value)

        self.capacity = capacity

        self.hot_cache = collections.OrderedDict()
        self.cold_store = {}

        self.promotions_to_hot = 0
        self.demotions_to_cold = 0

    def _touch(self, key):
        if key in self.hot_cache:
            self.hot_cache.move_to_end(key)
            return

        if key in self.cold_store:
            value = self.cold_store.pop(key)
            self.promotions_to_hot += 1
        else:
            value = self._default_factory()
            self.promotions_to_hot += 1

        self.hot_cache[key] = value
        self.hot_cache.move_to_end(key)

        if len(self.hot_cache) > self.capacity:
            old_key, old_val = self.hot_cache.popitem(last=False)
            self.cold_store[old_key] = old_val
            self.demotions_to_cold += 1


    def get(self, state: Hashable, action: Hashable) -> float:
        key = (state, action)

        self._touch(key)
        return self.hot_cache[key]

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        key = (state, action)

        self._touch(key)
        self.hot_cache[key] = value

    def size(self):
        return len(self.hot_cache) + len(self.cold_store)

    def exists(self, state: Hashable, action: Hashable) -> bool:
        key = (state, action)
        return key in self.hot_cache or key in self.cold_store

    def dump_all(self):
        combined = self.cold_store.copy()
        combined.update(self.hot_cache)
        return combined
    
    def reset_stats(self):
        self.promotions_to_hot = 0
        self.demotions_to_cold = 0

    def get_stats(self):
        return {
            "promotions_to_hot": self.promotions_to_hot,
            "demotions_to_cold": self.demotions_to_cold,
        }

