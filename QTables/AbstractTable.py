from abc import ABC, abstractmethod
from typing import Any, Dict, Hashable, Iterable, Tuple, Optional, Callable



Key = Tuple[Hashable, Hashable]  # (state, action)


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
            self._default_factory: Callable[[], float] = default_value  # type: ignore
        else:
            self._default_factory = lambda: default_value  # type: ignore

    @abstractmethod
    def get(self, state: Hashable, action: Hashable) -> float:
        """Return the Q-value for (state, action) or the default if unseen."""

    @abstractmethod
    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        """Set the Q-value for (state, action)."""