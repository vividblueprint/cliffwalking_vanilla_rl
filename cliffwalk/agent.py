"""A small epsilon-greedy Q-learning agent."""

from __future__ import annotations

import numpy as np

class QLearningAgent:
    def __init__(
        self,
        n_states: int,
        n_actions: int,
        learning_rate: float = 0.5,
        discount: float = 1.0,
        epsilon: float = 1.0,
        epsilon_min: float = 0.05,
        epsilon_decay: float = 0.995,
        seed: int | None = None,
    ) -> None:
        self.q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    def act(self, state: int) -> int:
        if self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        best_actions = np.flatnonzero(self.q[state] == self.q[state].max())
        return int(self.rng.choice(best_actions))

    def learn(self, state: int, action: int, reward: float, next_state: int, done: bool) -> None:
        target = reward if done else reward + self.discount * self.q[next_state].max()
        self.q[state, action] += self.learning_rate * (target - self.q[state, action])

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
