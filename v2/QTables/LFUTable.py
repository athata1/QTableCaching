import time
import heapq
from QTables.AbstractTable import AbstractQTable
from typing import Hashable, Optional, Tuple, Dict, List

Key = Tuple[Hashable, Hashable]


class LFUQTable(AbstractQTable):
    def __init__(self,
                 default_value: Optional[float] = 0.0,
                 hot_capacity: int = 1000,
                 move_every: int = 10000):
        super().__init__(default_value)
        self.hot_capacity = int(hot_capacity)
        self.move_every = int(move_every)

        self.hot_cache: Dict[Key, Tuple[float, int]] = {}
        self.cold_cache: Dict[Key, Tuple[float, int]] = {}

        self._heap: List[Tuple[int, float, Key]] = []
        self._access_counter = 0
        self._time_cache = time.time

        self.promotions_to_hot = 0
        self.demotions_to_cold = 0

    def reset_stats(self) -> None:
        """Called at the start of each episode."""
        self.promotions_to_hot = 0
        self.demotions_to_cold = 0

    def get_stats(self) -> Dict[str, float]:
        """Called at the end of each episode."""
        return {
            "promotions_to_hot": self.promotions_to_hot,
            "demotions_to_cold": self.demotions_to_cold,
        }

    def _promote_to_hot(self, key: Key, value: float, freq: int) -> None:
        """Move an entry into the hot cache and count a promotion."""
        self.hot_cache[key] = (value, freq)
        self.promotions_to_hot += 1

    def _rebalance(self) -> None:
        if len(self.hot_cache) <= self.hot_capacity:
            return

        now = self._time_cache()
        self._heap = [
            (freq, now, key) for key, (val, freq) in self.hot_cache.items()
        ]
        heapq.heapify(self._heap)

        excess = len(self.hot_cache) - self.hot_capacity
        for _ in range(excess):
            if not self._heap:
                break
            freq, _, key = heapq.heappop(self._heap)
            val, _ = self.hot_cache.pop(key, (None, None))
            if val is not None:
                self.cold_cache[key] = (val, freq)
                self.demotions_to_cold += 1

    def get(self, state: Hashable, action: Hashable) -> float:
        key: Key = (state, action)
        self._access_counter += 1

        if key in self.hot_cache:
            val, freq = self.hot_cache[key]
            self.hot_cache[key] = (val, freq + 1)
        elif key in self.cold_cache:
            val, freq = self.cold_cache.pop(key)
            self._promote_to_hot(key, val, freq + 1)
        else:
            val = self._default_factory()
            self._promote_to_hot(key, val, 1)

        if self._access_counter >= self.move_every:
            self._rebalance()
            self._access_counter = 0

        return val

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        key: Key = (state, action)
        self._access_counter += 1

        if key in self.hot_cache:
            _, freq = self.hot_cache[key]
            self.hot_cache[key] = (value, freq + 1)
        elif key in self.cold_cache:
            _, freq = self.cold_cache.pop(key)
            self._promote_to_hot(key, value, freq + 1)
        else:
            self._promote_to_hot(key, value, 1)

        if self._access_counter >= self.move_every:
            self._rebalance()
            self._access_counter = 0
