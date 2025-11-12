import collections
from typing import Hashable, Tuple, Dict
from QTables.AbstractTable import AbstractQTable

class LRUQTable(AbstractQTable):
    """
    A Q-table with a dual storage system:
      - hot_cache: stores the most recently used entries up to `capacity`
      - cold_store: stores every other entry not in hot_cache

    No values are ever lost: if LRU pushes out from hot, it's moved to cold.
    """

    def __init__(self, capacity: int, default_value: float = 0.0):
        super().__init__(default_value)

        # Capacity only affects hot storage
        self.capacity = capacity

        # OrderedDict keeps insertion order so we can pop LRU
        self.hot_cache = collections.OrderedDict()  # {(state, action): value}
        self.cold_store = {}                        # {(state, action): value}

    def _touch(self, key: Tuple[Hashable, Hashable]) -> None:
        """
        Move key into the hot cache, updating recency.
        If hot exceeds capacity, move LRU item into cold storage.
        """
        if key in self.hot_cache:
            # Move to end (most recent)
            self.hot_cache.move_to_end(key)
            return

        # Key not in hot - check if it exists in cold store
        if key in self.cold_store:
            value = self.cold_store.pop(key)
        else:
            # Brand new key → default initialization
            value = self._default_factory()

        # Insert into hot cache
        self.hot_cache[key] = value
        self.hot_cache.move_to_end(key)

        # Enforce LRU capacity
        if len(self.hot_cache) > self.capacity:
            old_key, old_val = self.hot_cache.popitem(last=False)  # pop LRU
            self.cold_store[old_key] = old_val

    def get(self, state: Hashable, action: Hashable) -> float:
        key = (state, action)

        # Ensure key is considered "used"
        self._touch(key)
        return self.hot_cache[key]

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        key = (state, action)

        # Ensure key is moved to hot & then update value
        self._touch(key)
        self.hot_cache[key] = value

    def size(self):
        """
        Total entries stored across both hot and cold.
        """
        return len(self.hot_cache) + len(self.cold_store)

    def exists(self, state: Hashable, action: Hashable) -> bool:
        key = (state, action)
        return key in self.hot_cache or key in self.cold_store

    def dump_all(self):
        """
        Returns a dict of all Q-values merged (hot overrides cold where needed).
        """
        combined = self.cold_store.copy()
        combined.update(self.hot_cache)
        return combined
