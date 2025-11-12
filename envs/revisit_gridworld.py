import gymnasium as gym
from gymnasium import spaces
import numpy as np

class DenseRevisitGridWorldEnv(gym.Env):
    def __init__(self, width=3, height=3, max_steps=50, revisit_penalty=-0.1):
        super().__init__()
        self.width = width
        self.height = height
        self.max_steps = max_steps
        self.revisit_penalty = revisit_penalty
        self.observation_space = gym.spaces.Discrete(width * height)
        self.action_space = gym.spaces.Discrete(4)  # up, down, left, right
        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.agent_pos = 0
        self.goal_pos = self.width * self.height - 1
        self.steps = 0
        self.visited = {self.agent_pos}  # track visited states
        return self._get_obs(), {}

    def _get_obs(self):
        return self.agent_pos

    def step(self, action):
        x, y = divmod(self.agent_pos, self.width)

        if action == 0: y = max(0, y - 1)     # up
        elif action == 1: y = min(self.height - 1, y + 1)  # down
        elif action == 2: x = max(0, x - 1)   # left
        elif action == 3: x = min(self.width - 1, x + 1)   # right

        new_pos = x + y * self.width

        # Base reward per step
        reward = -0.01

        # Check revisit
        if new_pos in self.visited:
            reward += self.revisit_penalty   # add punishment
        else:
            self.visited.add(new_pos)

        self.agent_pos = new_pos
        self.steps += 1

        # Terminal checks
        done = self.agent_pos == self.goal_pos or self.steps >= self.max_steps

        if self.agent_pos == self.goal_pos:
            reward = 1.0
        elif done and self.agent_pos != self.goal_pos:
            reward = 0.0

        return self._get_obs(), reward, done, False, {}
