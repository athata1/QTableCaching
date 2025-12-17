import gymnasium as gym
import random
import time
from typing import Tuple
from QTables.ArrayTable import ArrayQTable
from QTables.DictTable import DictQTable
from QTables.LRUTable import LRUQTable
from QTables.TimestampTable import TimestampQTable
from QTables.LFUTable import LFUQTable
from envs.gridworld import DenseGridWorldEnv
from envs.revisit_gridworld import DenseRevisitGridWorldEnv
import random
import numpy as np

# Q-learning parameters
alpha = 0.1         # learning rate
gamma = 0.99        # discount factor
epsilon = 1.0       # initial exploration rate
epsilon_min = 0.1   # minimum exploration
episodes = 40000    # total training episodes
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = (epsilon_end / epsilon_start) ** (1 / episodes)

# Environment parameters
grid_width = 50
grid_height = 50
max_steps = 2**(grid_width + grid_height) # max steps per episode

# Create DenseGridWorld environment
env = DenseGridWorldEnv(width=grid_width, height=grid_height, max_steps=max_steps)

num_states = env.observation_space.n
num_actions = env.action_space.n

# Initialize Q-table
#q_table = LRUQTable(capacity=num_states * num_actions * 0.7, default_value=0.0)
#q_table = DictQTable(default_value=0.0)
#q_table = ArrayQTable(num_states, num_actions, default_value=0.0)
q_table = LFUQTable(hot_capacity=int(num_states * num_actions * 0.7), default_value=0.0, move_every=20000)
#q_table = TimestampQTable(default_value=0.0, hot_duration=10.0, move_every=10000)

goal_pos = env.goal_pos  # DenseGridWorld has a single goal

# Start training timer
start_time = time.time()
print("Training started...")

# Training loop
for ep in range(episodes):
    state, _ = env.reset(seed=ep)
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

        # Simple reward: +1 for reaching goal, 0 otherwise
        reward = 1.0 if next_state == goal_pos else 0.0

        # Q-learning update
        max_next_q = max(q_table.get(next_state, a) for a in range(num_actions))
        new_q = q_table.get(state, action) + alpha * (reward + gamma * max_next_q - q_table.get(state, action))
        q_table.set(state, action, new_q)

        state = next_state
        if done:
            break

    # Decay epsilon
    if epsilon > epsilon_min:
        epsilon *= epsilon_decay

    # Print progress every 500 episodes
    if (ep + 1) % 500 == 0:
        print(f"Episode {ep+1}/{episodes}, epsilon: {epsilon:.3f}")

# End training timer
end_time = time.time()
print(f"Training finished in {end_time - start_time:.2f} seconds.")

# Evaluation
wins = 0
eval_episodes = 1000
for _ in range(eval_episodes):
    state, _ = env.reset()
    state = int(state)
    done = False
    while not done:
        action = max(range(num_actions), key=lambda a: q_table.get(state, a))
        action = int(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = int(next_state)
        done = terminated or truncated

        reward = 1.0 if next_state == goal_pos else 0.0

        state = next_state
    wins += reward > 0  # count as success if agent reaches goal

print(f"Success rate over {eval_episodes} episodes: {wins / eval_episodes:.2f}")
