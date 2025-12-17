import gymnasium as gym
from gymnasium.envs.toy_text.frozen_lake import generate_random_map
import random
import numpy as np
import time  # <-- import time
from typing import Hashable, Tuple
from QTables.AbstractTable import AbstractQTable
from QTables.ArrayTable import ArrayQTable
from QTables.DictTable import DictQTable

# Q-learning parameters
alpha = 0.1         # learning rate
gamma = 0.99        # discount factor
epsilon = 1.0       # initial exploration rate
epsilon_min = 0.1   # minimum exploration
episodes = 40000    # total training episodes
epsilon_start = 1.0
epsilon_end = 0.1
epsilon_decay = (epsilon_end / epsilon_start) ** (1 / episodes)
max_steps = 100      # max steps per episode

# Set seeds for reproducibility
#random.seed(42)
#np.random.seed(42)

map_size = 20
print("Generating random map...")
random_map = generate_random_map(size=map_size)
print("\n".join(random_map))

# Create environment
env = gym.make("FrozenLake-v1", desc=random_map, is_slippery=True)

num_states = env.observation_space.n
num_actions = env.action_space.n

# Initialize Q-table
q_table = ArrayQTable(num_states, num_actions, default_value=0.0)
#q_table = DictQTable(default_value=0.0)

# Helper functions
def get_coords(pos: int, width: int) -> Tuple[int, int]:
    """Convert linear state index to (row, col)."""
    return divmod(pos, width)

def manhattan_distance(pos1: int, pos2: int, width: int) -> int:
    x1, y1 = get_coords(pos1, width)
    x2, y2 = get_coords(pos2, width)
    return abs(x1 - x2) + abs(y1 - y2)

# Precompute goal and hole positions
goal_pos = None
hole_positions = set()
for i, row in enumerate(random_map):
    for j, cell in enumerate(row):
        state = i * map_size + j
        if cell == "G":
            goal_pos = state
        elif cell == "H":
            hole_positions.add(state)

# Start training timer
start_time = time.time()
print("Training started...")

# Training loop
for ep in range(episodes):
    state, _ = env.reset(seed=ep)
    state = int(state)
    prev_distance = manhattan_distance(state, goal_pos, map_size)
    
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

        # --- DENSE REWARD ---
        if next_state == goal_pos:
            reward = 1.0
        elif next_state in hole_positions:
            reward = -1.0
        else:
            current_distance = manhattan_distance(next_state, goal_pos, map_size)
            reward = prev_distance - current_distance  # reward for moving closer
            prev_distance = current_distance

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
    prev_distance = manhattan_distance(state, goal_pos, map_size)
    done = False
    while not done:
        action = max(range(num_actions), key=lambda a: q_table.get(state, a))
        action = int(action)
        next_state, reward, terminated, truncated, info = env.step(action)
        next_state = int(next_state)
        done = terminated or truncated

        # Use same dense reward for evaluation
        if next_state == goal_pos:
            reward = 1.0
        elif next_state in hole_positions:
            reward = -1.0
        else:
            current_distance = manhattan_distance(next_state, goal_pos, map_size)
            reward = prev_distance - current_distance
            prev_distance = current_distance

        state = next_state
    wins += reward > 0  # count as success if agent reaches goal

print(f"Success rate over {eval_episodes} episodes: {wins / eval_episodes:.2f}")
