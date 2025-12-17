import gymnasium as gym
import random
import numpy as np
import time
import csv

from QTables.ArrayTable import ArrayQTable
from QTables.DictTable import DictQTable
from QTables.LRUTable import LRUQTable
from QTables.LFUTable import LFUQTable
from QTables.TimestampTable import TimestampQTable

# Q-learning parameters (fixed)
alpha = 0.1
gamma = 0.99
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_min = 0.1
episodes = 40000
max_steps = 100
eval_episodes = 1000
epsilon_decay = (epsilon_end / epsilon_start) ** (1 / episodes)

table_configs = [

    ("array", {}),
    ("array", {}),

    ("dict", {}),
    ("dict", {}),

    ("lru", {"capacity": 200}),
    ("lru", {"capacity": 400}),
    ("lru", {"capacity": 800}),
    ("lru", {"capacity": 1200}),
    ("lru", {"capacity": 1600}),
    ("lru", {"capacity": 2000}),
    ("lru", {"capacity": 50}),
    ("lru", {"capacity": 3000}),

    ("lfu", {"hot_capacity": 200, "move_every": 2000}),
    ("lfu", {"hot_capacity": 300, "move_every": 5000}),
    ("lfu", {"hot_capacity": 800, "move_every": 5000}),
    ("lfu", {"hot_capacity": 1000, "move_every": 10000}),
    ("lfu", {"hot_capacity": 1500, "move_every": 5000}),
    ("lfu", {"hot_capacity": 2000, "move_every": 10000}),
    ("lfu", {"hot_capacity": 50, "move_every": 1000}),
    ("lfu", {"hot_capacity": 3000, "move_every": 20000}),

    ("timestamp", {"hot_duration": 0.5, "move_every": 2000}),
    ("timestamp", {"hot_duration": 1.0, "move_every": 5000}),
    ("timestamp", {"hot_duration": 3.0, "move_every": 5000}),
    ("timestamp", {"hot_duration": 5.0, "move_every": 10000}),
    ("timestamp", {"hot_duration": 10.0, "move_every": 10000}),
    ("timestamp", {"hot_duration": 15.0, "move_every": 20000}),
]

csv_file = "CWresults.csv"
writer = csv.DictWriter(
    open(csv_file, "w", newline=""),
    fieldnames=[
        "table_type", "capacity", "hot_capacity", "move_every", "hot_duration",
        "success_rate", "train_time_sec"
    ]
)
writer.writeheader()

print(f"Running {len(table_configs)} experiments on CliffWalking-v0...\n")

for table_type, params in table_configs:

    print("\n--------------------------------------------")
    print(f"Running table: {table_type}, params={params}")
    print("--------------------------------------------")

    epsilon = epsilon_start

    env = gym.make("CliffWalking-v0")

    num_states = env.observation_space.n
    num_actions = env.action_space.n
    goal_state = num_states - 1  # in CliffWalking, goal is usually last state (bottom-right)

    if table_type == "array":
        q_table = ArrayQTable(num_states, num_actions, default_value=0.0)
    elif table_type == "dict":
        q_table = DictQTable(default_value=0.0)
    elif table_type == "lru":
        q_table = LRUQTable(**params)
    elif table_type == "lfu":
        q_table = LFUQTable(**params)
    elif table_type == "timestamp":
        q_table = TimestampQTable(**params)
    else:
        raise ValueError("Unknown table type")

    start_time = time.time()
    print("Training started...")

    for ep in range(episodes):
        state, _ = env.reset(seed=ep)
        state = int(state)

        for t in range(max_steps):
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = max(range(num_actions), key=lambda a: q_table.get(state, a))
            action = int(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_state)
            done = terminated or truncated

            # Use environment reward directly (CliffWalking: -1 each step, -100 on cliff)
            max_next_q = max(q_table.get(next_state, a) for a in range(num_actions))
            new_q = q_table.get(state, action) + alpha * (reward + gamma * max_next_q - q_table.get(state, action))
            q_table.set(state, action, new_q)

            state = next_state
            if done:
                break

        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

        if (ep + 1) % 500 == 0:
            print(f"Episode {ep+1}/{episodes}, epsilon: {epsilon:.3f}")

    train_time = time.time() - start_time
    print(f"Training finished in {train_time:.2f} seconds.")

    wins = 0
    for _ in range(eval_episodes):
        state, _ = env.reset()
        state = int(state)
        done = False

        while not done:
            action = max(range(num_actions), key=lambda a: q_table.get(state, a))
            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_state)
            done = terminated or truncated
            state = next_state

        if state == goal_state:
            wins += 1

    success_rate = wins / eval_episodes
    print(f"Success rate: {success_rate:.2f}")

    # Write CSV row
    row = {
        "table_type": table_type,
        "capacity": params.get("capacity"),
        "hot_capacity": params.get("hot_capacity"),
        "move_every": params.get("move_every"),
        "hot_duration": params.get("hot_duration"),
        "success_rate": success_rate,
        "train_time_sec": round(train_time, 3),
    }
    writer.writerow(row)

print("\nAll CliffWalking experiments finished. Results saved to CWresults.csv\n")
