import time
import heapq
from QTables.AbstractTable import AbstractQTable
from typing import Hashable, Optional, Tuple, Dict, List

Key = Tuple[Hashable, Hashable]  # (state, action)


class LFUQTable(AbstractQTable):
    """
    Optimized LFU-based Q-table with hot and cold caches.
    Uses a min-heap for efficient demotion of least-frequently-used entries.
    """

    def __init__(self, 
                 default_value: Optional[float] = 0.0, 
                 hot_capacity: int = 1000, 
                 move_every: int = 10000):
        super().__init__(default_value)
        self.hot_capacity = hot_capacity
        self.move_every = move_every

        # hot_cache maps (state, action) -> (value, freq)
        self.hot_cache: Dict[Key, Tuple[float, int]] = {}
        # cold_cache maps (state, action) -> (value, freq)
        self.cold_cache: Dict[Key, Tuple[float, int]] = {}

        # Heap entries: (freq, last_access_time, key)
        self._heap: List[Tuple[int, float, Key]] = []
        self._access_counter = 0
        self._time_cache = time.time  # micro-optimization

    def _rebalance(self):
        """Demote least frequently used entries if hot cache exceeds capacity."""
        if len(self.hot_cache) <= self.hot_capacity:
            return

        now = self._time_cache()
        # Rebuild heap once for accuracy (lazy invalidation)
        self._heap.clear()
        for k, (_, freq) in self.hot_cache.items():
            self._heap.append((freq, now, k))
        heapq.heapify(self._heap)

        # Move least-frequent items to cold cache
        excess = len(self.hot_cache) - self.hot_capacity
        for _ in range(excess):
            if not self._heap:
                break
            freq, _, key = heapq.heappop(self._heap)
            val, _ = self.hot_cache.pop(key, (None, None))
            if val is not None:
                self.cold_cache[key] = (val, freq)

    def get(self, state: Hashable, action: Hashable) -> float:
        key = (state, action)
        self._access_counter += 1

        if key in self.hot_cache:
            val, freq = self.hot_cache[key]
            self.hot_cache[key] = (val, freq + 1)
        elif key in self.cold_cache:
            val, freq = self.cold_cache.pop(key)
            self.hot_cache[key] = (val, freq + 1)
        else:
            val = self._default_factory()
            self.cold_cache[key] = (val, 1)

        # Trigger periodic rebalance
        if self._access_counter >= self.move_every:
            self._rebalance()
            self._access_counter = 0

        return val

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        key = (state, action)
        self._access_counter += 1

        if key in self.hot_cache:
            _, freq = self.hot_cache[key]
            self.hot_cache[key] = (value, freq + 1)
        elif key in self.cold_cache:
            _, freq = self.cold_cache.pop(key)
            self.hot_cache[key] = (value, freq + 1)
        else:
            self.hot_cache[key] = (value, 1)

        # Trigger periodic rebalance
        if self._access_counter >= self.move_every:
            self._rebalance()
            self._access_counter = 0
