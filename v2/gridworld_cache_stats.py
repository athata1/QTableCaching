import random
import time
import csv

import numpy as np

from QTables.LRUTable import LRUQTable
from QTables.TimestampTable import TimestampQTable
from QTables.LFUTable import LFUQTable
from envs.revisit_gridworld import DenseRevisitGridWorldEnv

GLOBAL_SEED = 123
random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)

ALPHA = 0.1
GAMMA = 0.99
EPISODES = 5000
EPSILON_START = 1.0
EPSILON_END = 0.1

GRID_SIZE = 50
MAX_STEPS_FACTOR = 4


def make_env():
    width = GRID_SIZE
    height = GRID_SIZE
    max_steps = MAX_STEPS_FACTOR * width * height
    env = DenseRevisitGridWorldEnv(width=width, height=height, max_steps=max_steps)
    return env


def make_q_table(name, env, hot_fraction=0.7):
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    total_sa = num_states * num_actions

    if name == "lru":
        return LRUQTable(capacity=int(hot_fraction * total_sa), default_value=0.0)
    if name == "lfu":
        return LFUQTable(
            hot_capacity=int(hot_fraction * total_sa),
            default_value=0.0,
            move_every=20000,
        )
    if name == "timestamp":
        return TimestampQTable(
            default_value=0.0,
            hot_duration=10.0,
            move_every=10000,
        )
    raise ValueError(f"Unknown table: {name}")


def train_with_stats(env, q_table, table_name, csv_writer):
    num_actions = env.action_space.n
    epsilon = EPSILON_START
    epsilon_decay = (EPSILON_END / EPSILON_START) ** (1.0 / EPISODES)
    goal_pos = env.goal_pos

    for ep in range(EPISODES):
        q_table.reset_stats()

        state, _ = env.reset(seed=ep)
        state = int(state)
        done = False
        t = 0

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
            new_q = old_q + ALPHA * (reward + GAMMA * max_next_q - old_q)
            q_table.set(state, action, new_q)

            state = next_state
            t += 1

        if epsilon > EPSILON_END:
            epsilon *= epsilon_decay

        stats = q_table.get_stats()
        csv_writer.writerow([
            table_name,
            ep,
            t,
            stats.get("promotions_to_hot", 0),
            stats.get("demotions_to_cold", 0),
        ])


if __name__ == "__main__":
    env = make_env()
    output_path = "cache_stats_exp3.csv"

    table_names = ["lru", "lfu", "timestamp"]

    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "table",
            "episode",
            "episode_length",
            "promotions_to_hot",
            "demotions_to_cold",
        ])

        for name in table_names:
            print(f"Running cache stats experiment for table: {name}")
            q_table = make_q_table(name, env, hot_fraction=0.7)
            train_with_stats(env, q_table, name, writer)

    print(f"Saved cache stats to {output_path}")
