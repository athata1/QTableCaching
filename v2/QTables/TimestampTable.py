import time
from QTables.AbstractTable import AbstractQTable
from typing import Hashable, Callable, Optional, Tuple, Dict

Key = Tuple[Hashable, Hashable]

class TimestampQTable(AbstractQTable):
    def __init__(self, default_value: Optional[float] = 0.0, hot_duration: float = 10.0, move_every: int = 10000):
        """
        default_value: returned when a Q-value is unseen.
        hot_duration: time in seconds before a hot entry moves to cold cache.
        move_every: number of sets before moving hot entries to cold cache.
        """
        super().__init__(default_value)
        self.hot_duration = hot_duration
        self.move_every = move_every
        self.hot_cache: Dict[Key, Tuple[float, float]] = {}
        self.cold_cache: Dict[Key, float] = {}
        self._set_counter = 0

        self.promotions_to_hot = 0
        self.demotions_to_cold = 0

    def _move_hot_to_cold(self):
        now = time.time()
        to_move = []
        for key, (val, ts) in self.hot_cache.items():
            if now - ts >= self.hot_duration:
                to_move.append(key)
        for key in to_move:
            val, ts = self.hot_cache.pop(key)
            self.cold_cache[key] = val
            self.demotions_to_cold += 1


    def get(self, state: Hashable, action: Hashable) -> float:
        key = (state, action)
        if key in self.hot_cache:
            value, _ = self.hot_cache[key]
            self.hot_cache[key] = (value, time.time())
            return value
        if key in self.cold_cache:
            return self.cold_cache[key]
        return self._default_factory()

    def set(self, state, action, value):
        key = (state, action)
        if key in self.cold_cache:
            self.cold_cache.pop(key)
            self.promotions_to_hot += 1
        else:
            if key not in self.hot_cache:
                self.promotions_to_hot += 1

        self.hot_cache[key] = (value, time.time())

        self._set_counter += 1
        if self._set_counter >= self.move_every:
            self._move_hot_to_cold()
            self._set_counter = 0
    def reset_stats(self):
        self.promotions_to_hot = 0
        self.demotions_to_cold = 0

    def get_stats(self):
        return {
            "promotions_to_hot": self.promotions_to_hot,
            "demotions_to_cold": self.demotions_to_cold,
        }

