import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random
import numpy as np
import time
from QTables.ArrayTable import ArrayQTable
from QTables.DictTable import DictQTable

# Q-learning parameters
alpha = 0.1
gamma = 0.99
epsilon_start = 1.0
epsilon_min = 0.1
epsilon_decay = 0.999
episodes = 5000
max_steps = 100

# Number of training runs to sample
n_runs = 5
training_times = []

# Environment parameters
map_size = 20
random_map = generate_random_map(size=map_size)
print("Random map:\n" + "\n".join(random_map))

for run in range(n_runs):
    print(f"\n=== Training run {run+1}/{n_runs} ===")
    env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=True)
    num_states = env.observation_space.n
    num_actions = env.action_space.n
    #q_table = ArrayQTable(num_states, num_actions, default_value=0.0)
    q_table = DictQTable(default_value=0.0)

    epsilon = epsilon_start

    start_time = time.time()

    # Training loop
    for ep in range(episodes):
        state, _ = env.reset()
        state = int(state)

        for t in range(max_steps):
            # Epsilon-greedy action selection
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = max(range(num_actions), key=lambda a: q_table.get(state, a))
            action = int(action)

            next_state, reward, terminated, truncated, info = env.step(action)
            next_state = int(next_state)
            done = terminated or truncated

            max_next_q = max(q_table.get(next_state, a) for a in range(num_actions))
            new_q = q_table.get(state, action) + alpha * (reward + gamma * max_next_q - q_table.get(state, action))
            q_table.set(state, action, new_q)

            state = next_state
            if done:
                break

        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay

    end_time = time.time()
    elapsed = end_time - start_time
    training_times.append(elapsed)
    print(f"Run {run+1} training time: {elapsed:.2f} sec")

# Compute average
avg_time = sum(training_times) / n_runs
print(f"\nAverage training time over {n_runs} runs: {avg_time:.2f} sec")
