from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Iterable, Tuple, Optional, Callable


Key = Tuple[Hashable, Hashable]


class AbstractQTable(ABC):
    """
    Minimal abstract Q-table interface.
    Implementations must provide get and set for (state, action) pairs.
    """

    def __init__(self, default_value: Optional[float] = 0.0) -> None:
        """
        default_value: value returned by get(...) when an entry has not been set.
                       It may be a float or a zero-argument callable that returns a float.
        """
        if callable(default_value):
            self._default_factory: Callable[[], float] = default_value
        else:
            self._default_factory = lambda: default_value

    @abstractmethod
    def get(self, state: Hashable, action: Hashable) -> float:
        """Return the Q-value for (state, action) or the default if unseen."""

    @abstractmethod
    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        """Set the Q-value for (state, action)."""

    def reset_stats(self):
        """Reset per-episode statistics (promotions/demotions etc.)."""
        pass

    def get_stats(self):
        """
        Return a dict of stats for the current episode.
        At minimum, we standardize these keys:
          - 'promotions_to_hot'
          - 'demotions_to_cold'
        Non-cache tables can safely return zeros.
        """
        return {
            "promotions_to_hot": 0,
            "demotions_to_cold": 0,
        }