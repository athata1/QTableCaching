import numpy as np
from typing import Hashable, Tuple
from QTables.AbstractTable import AbstractQTable

class ArrayQTable(AbstractQTable):
    """
    Concrete Q-table implemented as a 2D NumPy array of shape
    (num_states, num_actions).
    """

    def __init__(self, num_states: int, num_actions: int, default_value: float = 0.0):
        super().__init__(default_value)
        self.num_states = num_states
        self.num_actions = num_actions

        # Fill initial Q-values with the default
        self._table = np.full((num_states, num_actions), self._default_factory(), dtype=float)

    def _check_bounds(self, state: Hashable, action: Hashable) -> Tuple[int, int]:
        """Ensure state and action are valid integer indices."""
        if not isinstance(state, int) or not (0 <= state < self.num_states):
            raise ValueError(f"State {state} is out of bounds (0..{self.num_states - 1}).")
        if not isinstance(action, int) or not (0 <= action < self.num_actions):
            raise ValueError(f"Action {action} is out of bounds (0..{self.num_actions - 1}).")
        return state, action

    def get(self, state: Hashable, action: Hashable) -> float:
        s, a = self._check_bounds(state, action)
        return self._table[s, a]

    def set(self, state: Hashable, action: Hashable, value: float) -> None:
        s, a = self._check_bounds(state, action)
        self._table[s, a] = value
