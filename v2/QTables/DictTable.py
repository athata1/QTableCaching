from typing import Hashable, Dict, Tuple
from QTables.AbstractTable import AbstractQTable

class DictQTable(AbstractQTable):
    """
    Q-table backed by a Python dictionary:
        key = (state, action)
        value = float Q-value

    Unseen entries return the default value.
    """

    def __init__(self, default_value: float = 0.0):
        super().__init__(default_value)
        self._data: Dict[Tuple[Hashable, Hashable], float] = {}

    def get(self, state: Hashable, action: Hashable) -> float:
        return self._data.get((state, action), self._default_factory())

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        self._data[(state, action)] = value
    
    def reset_stats(self):
        pass

    def get_stats(self):
        return {
            "promotions_to_hot": 0,
            "demotions_to_cold": 0,
        }
