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


def make_env(grid_size: int, use_revisit: bool = True):
    """
    Create an environment of size grid_size x grid_size.

    By default uses DenseRevisitGridWorldEnv to stress caching behavior.
    Set use_revisit=False to use the plain DenseGridWorldEnv instead.
    """
    width = grid_size
    height = grid_size
    max_steps = 4 * width * height

    if use_revisit:
        env = DenseRevisitGridWorldEnv(width=width, height=height, max_steps=max_steps)
        env_type = "revisit"
    else:
        env = DenseGridWorldEnv(width=width, height=height, max_steps=max_steps)
        env_type = "dense"

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    return env, env_type, num_states, num_actions


def make_q_table(table_name: str, env, hot_fraction: float):
    """
    Construct a Q-table for the given table_name.

    For cache-based tables (LRU, LFU), hot_fraction controls the fraction
    of the full state-action space that is allowed in the "hot" cache.

    For baselines (array, dict, timestamp), hot_fraction is ignored.
    """
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    total_sa = num_states * num_actions

    if table_name == "array":
        return ArrayQTable(num_states, num_actions, default_value=0.0)

    if table_name == "dict":
        return DictQTable(default_value=0.0)

    if table_name == "lru":
        capacity = max(1, int(hot_fraction * total_sa))
        return LRUQTable(capacity=capacity, default_value=0.0)

    if table_name == "lfu":
        hot_capacity = max(1, int(hot_fraction * total_sa))
        return LFUQTable(
            hot_capacity=hot_capacity,
            default_value=0.0,
            move_every=20000,
        )

    if table_name == "timestamp":
        return TimestampQTable(
            default_value=0.0,
            hot_duration=10.0,
            move_every=10000,
        )

    raise ValueError(f"Unknown table type: {table_name}")


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

    Returns:
      {
        "success_rate": float,
        "train_time": float,
        "avg_eval_steps": float,
      }
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
    GRID_SIZE = 50

    HOT_FRACTIONS = [0.1, 0.3, 0.5, 0.7, 1.0]

    BASELINE_TABLES = ["array", "dict", "timestamp"]

    CAPACITY_TABLES = ["lru", "lfu"]

    env, env_type, num_states, num_actions = make_env(GRID_SIZE, use_revisit=True)

    output_path = "capacity_results.csv"

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "grid_size",
            "env_type",
            "table",
            "hot_fraction",
            "num_states",
            "num_actions",
            "success_rate",
            "train_time",
            "avg_eval_steps",
        ])

        print(f"Environment: {env_type}, size={GRID_SIZE}x{GRID_SIZE}, "
              f"states={num_states}, actions={num_actions}")

        for table_name in BASELINE_TABLES:
            print(f"\n[Baseline] Training table={table_name}")
            q_table = make_q_table(table_name, env, hot_fraction=1.0)

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
                f"  success_rate={metrics['success_rate']:.3f}, "
                f"train_time={metrics['train_time']:.1f}s, "
                f"avg_eval_steps={metrics['avg_eval_steps']:.1f}"
            )

            writer.writerow([
                GRID_SIZE,
                env_type,
                table_name,
                1.0,
                num_states,
                num_actions,
                f"{metrics['success_rate']:.4f}",
                f"{metrics['train_time']:.4f}",
                f"{metrics['avg_eval_steps']:.4f}",
            ])

        for hot_fraction in HOT_FRACTIONS:
            for table_name in CAPACITY_TABLES:
                print(f"\n[Capacity] table={table_name}, hot_fraction={hot_fraction}")
                q_table = make_q_table(table_name, env, hot_fraction=hot_fraction)

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
                    f"  success_rate={metrics['success_rate']:.3f}, "
                    f"train_time={metrics['train_time']:.1f}s, "
                    f"avg_eval_steps={metrics['avg_eval_steps']:.1f}"
                )

                writer.writerow([
                    GRID_SIZE,
                    env_type,
                    table_name,
                    hot_fraction,
                    num_states,
                    num_actions,
                    f"{metrics['success_rate']:.4f}",
                    f"{metrics['train_time']:.4f}",
                    f"{metrics['avg_eval_steps']:.4f}",
                ])

    print(f"\nSaved Experiment 2 results to {output_path}")
