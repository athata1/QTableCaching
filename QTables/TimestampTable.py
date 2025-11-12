import time
from QTables.AbstractTable import AbstractQTable
from typing import Hashable, Callable, Optional, Tuple, Dict

Key = Tuple[Hashable, Hashable]  # (state, action)

class TimestampQTable(AbstractQTable):
    """
    Q-table with hot/cold cache system based on timestamps.
    """

    def __init__(self, default_value: Optional[float] = 0.0, hot_duration: float = 10.0, move_every: int = 10000):
        """
        default_value: returned when a Q-value is unseen.
        hot_duration: time in seconds before a hot entry moves to cold cache.
        move_every: number of sets before moving hot entries to cold cache.
        """
        super().__init__(default_value)
        self.hot_duration = hot_duration
        self.move_every = move_every
        self.hot_cache: Dict[Key, Tuple[float, float]] = {}  # key -> (value, timestamp)
        self.cold_cache: Dict[Key, float] = {}  # key -> value
        self._set_counter = 0

    def _move_hot_to_cold(self):
        """Move expired hot cache items to cold cache."""
        now = time.time()
        expired_keys = [k for k, (_, ts) in self.hot_cache.items() if now - ts >= self.hot_duration]
        for k in expired_keys:
            value, _ = self.hot_cache.pop(k)
            self.cold_cache[k] = value

    def get(self, state: Hashable, action: Hashable) -> float:
        key = (state, action)
        # If in hot cache, update timestamp to keep it hot
        if key in self.hot_cache:
            value, _ = self.hot_cache[key]
            self.hot_cache[key] = (value, time.time())  # refresh timestamp
            return value
        if key in self.cold_cache:
            return self.cold_cache[key]
        return self._default_factory()

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        key = (state, action)
        self.hot_cache[key] = (value, time.time())
        if key in self.cold_cache:
            del self.cold_cache[key]

        # Increment set counter and conditionally move hot -> cold
        self._set_counter += 1
        if self._set_counter >= self.move_every:
            self._move_hot_to_cold()
            self._set_counter = 0
