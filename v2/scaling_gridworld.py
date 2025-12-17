import gymnasium as gym
import random
import time
import csv
from typing import Tuple

import numpy as np

from QTables.ArrayTable import ArrayQTable
from QTables.DictTable import DictQTable
from QTables.LRUTable import LRUQTable
from QTables.TimestampTable import TimestampQTable
from QTables.LFUTable import LFUQTable
from envs.gridworld import DenseGridWorldEnv
from envs.revisit_gridworld import DenseRevisitGridWorldEnv

GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

ALPHA = 0.1
GAMMA = 0.99
EPISODES = 40000
EVAL_EPISODES = 1000
EPSILON_START = 1.0
EPSILON_END = 0.1

HOT_FRACTION = 0.7

TABLE_FACTORIES = {
    "array": lambda env: ArrayQTable(
        env.observation_space.n,
        env.action_space.n,
        default_value=0.0,
    ),
    "dict": lambda env: DictQTable(
        default_value=0.0
    ),
    "lru": lambda env: LRUQTable(
        capacity=int(HOT_FRACTION * env.observation_space.n * env.action_space.n),
        default_value=0.0,
    ),
    "lfu": lambda env: LFUQTable(
        hot_capacity=int(HOT_FRACTION * env.observation_space.n * env.action_space.n),
        default_value=0.0,
        move_every=20000,
    ),
    "timestamp": lambda env: TimestampQTable(
        default_value=0.0,
        hot_duration=10.0,
        move_every=10000,
    ),
}


def make_env(grid_size: int) -> Tuple[DenseGridWorldEnv, int, int]:
    """
    Create a DenseGridWorldEnv of size grid_size x grid_size,
    with max_steps scaling with the grid area.
    """
    width = grid_size
    height = grid_size

    max_steps = 4 * width * height

    env = DenseGridWorldEnv(width=width, height=height, max_steps=max_steps)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    return env, num_states, num_actions


def train_and_evaluate(
    env,
    q_table,
    num_actions: int,
    episodes: int,
    eval_episodes: int,
    alpha: float,
    gamma: float,
    epsilon_start: float,
    epsilon_end: float,
):
    """
    Train Q-learning with the given Q-table on the given env,
    then evaluate greedily.

    Returns a dict with:
      - success_rate
      - train_time
      - avg_eval_steps
    """
    epsilon = epsilon_start
    epsilon_decay = (epsilon_end / epsilon_start) ** (1.0 / episodes)
    goal_pos = env.goal_pos

    start_time = time.time()

    for ep in range(episodes):
        state, _ = env.reset(seed=ep)
        state = int(state)

        done = False
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = max(range(num_actions), key=lambda a: q_table.get(state, a))
            action = int(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_state)
            done = terminated or truncated

            reward = 1.0 if next_state == goal_pos else 0.0

            max_next_q = max(q_table.get(next_state, a) for a in range(num_actions))
            old_q = q_table.get(state, action)
            new_q = old_q + alpha * (reward + gamma * max_next_q - old_q)
            q_table.set(state, action, new_q)

            state = next_state

        if epsilon > epsilon_end:
            epsilon *= epsilon_decay

    train_time = time.time() - start_time

    wins = 0
    total_steps = 0

    for ep in range(eval_episodes):
        state, _ = env.reset(seed=episodes + ep)
        state = int(state)
        done = False
        steps = 0

        while not done:
            action = max(range(num_actions), key=lambda a: q_table.get(state, a))
            action = int(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_state)
            done = terminated or truncated

            reward = 1.0 if next_state == goal_pos else 0.0

            state = next_state
            steps += 1

        wins += (reward > 0)
        total_steps += steps

    success_rate = wins / eval_episodes
    avg_steps = total_steps / eval_episodes if eval_episodes > 0 else float("nan")

    return {
        "success_rate": success_rate,
        "train_time": train_time,
        "avg_eval_steps": avg_steps,
    }


if __name__ == "__main__":
    GRID_SIZES = [10, 20, 30, 40, 50]
    TABLE_NAMES = ["array", "dict", "lru", "lfu", "timestamp"]

    output_path = "scaling_results.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "grid_size",
            "table",
            "num_states",
            "num_actions",
            "success_rate",
            "train_time",
            "avg_eval_steps",
        ])

        for grid_size in GRID_SIZES:
            env, num_states, num_actions = make_env(grid_size)

            print(f"\n=== Grid size: {grid_size}x{grid_size} (states={num_states}, actions={num_actions}) ===")

            for table_name in TABLE_NAMES:
                print(f"  -> Training with table: {table_name}")
                q_table = TABLE_FACTORIES[table_name](env)

                metrics = train_and_evaluate(
                    env=env,
                    q_table=q_table,
                    num_actions=num_actions,
                    episodes=EPISODES,
                    eval_episodes=EVAL_EPISODES,
                    alpha=ALPHA,
                    gamma=GAMMA,
                    epsilon_start=EPSILON_START,
                    epsilon_end=EPSILON_END,
                )

                print(
                    f"     success_rate={metrics['success_rate']:.3f}, "
                    f"train_time={metrics['train_time']:.1f}s, "
                    f"avg_eval_steps={metrics['avg_eval_steps']:.1f}"
                )

                writer.writerow([
                    grid_size,
                    table_name,
                    num_states,
                    num_actions,
                    f"{metrics['success_rate']:.4f}",
                    f"{metrics['train_time']:.4f}",
                    f"{metrics['avg_eval_steps']:.4f}",
                ])

    print(f"\nSaved scaling experiment results to {output_path}")
