import time
from QTables.AbstractTable import AbstractQTable
from typing import Hashable, Optional, Tuple, Dict

Key = Tuple[Hashable, Hashable]  # (state, action)


class LFUQTable(AbstractQTable):
    """
    Q-table using a Least-Frequently-Used (LFU) cache system with hot and cold layers.
    The hot cache keeps frequently accessed entries, while the cold cache holds infrequent ones.
    """

    def __init__(self, 
                 default_value: Optional[float] = 0.0, 
                 hot_capacity: int = 1000, 
                 move_every: int = 10000):
        """
        default_value: value returned when unseen (float or callable).
        hot_capacity: maximum size of the hot cache before demoting least-frequent entries.
        move_every: number of gets/sets before rebalancing hot/cold caches.
        """
        super().__init__(default_value)
        self.hot_capacity = hot_capacity
        self.move_every = move_every

        # hot_cache maps (state, action) -> (value, frequency, last_access_time)
        self.hot_cache: Dict[Key, Tuple[float, int, float]] = {}
        # cold_cache maps (state, action) -> (value, frequency)
        self.cold_cache: Dict[Key, Tuple[float, int]] = {}

        self._access_counter = 0  # counts both get and set calls

    def _rebalance(self):
        """
        Move least frequently used items from hot cache to cold cache if hot cache exceeds capacity.
        """
        if len(self.hot_cache) <= self.hot_capacity:
            return

        # Sort by (frequency, last_access_time) ascending
        sorted_hot = sorted(self.hot_cache.items(), key=lambda item: (item[1][1], item[1][2]))
        num_to_move = len(self.hot_cache) - self.hot_capacity

        for i in range(num_to_move):
            key, (value, freq, _) = sorted_hot[i]
            self.hot_cache.pop(key, None)
            self.cold_cache[key] = (value, freq)

    def _tick(self):
        """Increment operation counter and trigger rebalance if needed."""
        self._access_counter += 1
        if self._access_counter >= self.move_every:
            self._rebalance()
            self._access_counter = 0

    def get(self, state: Hashable, action: Hashable) -> float:
        key = (state, action)
        now = time.time()

        if key in self.hot_cache:
            value, freq, _ = self.hot_cache[key]
            self.hot_cache[key] = (value, freq + 1, now)
            self._tick()
            return value

        if key in self.cold_cache:
            value, freq = self.cold_cache.pop(key)
            # Promote to hot cache on access
            self.hot_cache[key] = (value, freq + 1, now)
            self._tick()
            return value

        # Unseen entry
        value = self._default_factory()
        self.cold_cache[key] = (value, 1)
        self._tick()
        return value

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        key = (state, action)
        now = time.time()

        # Update or insert into hot cache
        if key in self.hot_cache:
            _, freq, _ = self.hot_cache[key]
            self.hot_cache[key] = (value, freq + 1, now)
        elif key in self.cold_cache:
            _, freq = self.cold_cache.pop(key)
            self.hot_cache[key] = (value, freq + 1, now)
        else:
            self.hot_cache[key] = (value, 1, now)

        self._tick()
