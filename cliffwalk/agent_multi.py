"""Tabular control agents for comparing common RL update rules."""

from __future__ import annotations

import numpy as np


class _TabularAgent:
    """Shared epsilon-greedy policy and Q-table helpers."""

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
        if n_states < 1 or n_actions < 1:
            raise ValueError("n_states and n_actions must be positive")
        if not 0 < learning_rate <= 1:
            raise ValueError("learning_rate must be in (0, 1]")
        if not 0 <= discount <= 1:
            raise ValueError("discount must be in [0, 1]")
        if not 0 <= epsilon_min <= epsilon <= 1 or not 0 < epsilon_decay <= 1:
            raise ValueError("invalid epsilon schedule")
        self._q = np.zeros((n_states, n_actions), dtype=np.float32)
        self.learning_rate = learning_rate
        self.discount = discount
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.rng = np.random.default_rng(seed)

    @property
    def q(self) -> np.ndarray:
        return self._q

    @q.setter
    def q(self, value: np.ndarray) -> None:
        self._q = value

    def act(self, state: int, explore: bool = True) -> int:
        if explore and self.rng.random() < self.epsilon:
            return int(self.rng.integers(self.q.shape[1]))
        values = self.q[state]
        best_actions = np.flatnonzero(values == values.max()) # return the indices of nonzero elements
        return int(self.rng.choice(best_actions))

    def decay_exploration(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def learn(
        self,
        state: int,
        action: int,
        reward: float,
        next_state: int,
        done: bool,
        next_action: int | None = None,
    ) -> None:
        raise NotImplementedError


class QLearningAgent(_TabularAgent):
    """Off-policy one-step Q-learning."""

    def learn(self, state, action, reward, next_state, done, next_action=None) -> None:
        del next_action
        target = reward if done else reward + self.discount * float(self.q[next_state].max())
        self.q[state, action] += self.learning_rate * (target - self.q[state, action])


class DoubleQLearningAgent(_TabularAgent):
    """Double Q-learning, which reduces maximization overestimation."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.q_a = self._q.copy()
        self.q_b = self._q.copy()

    @property
    def q(self) -> np.ndarray:
        """Mean table, useful for greedy action selection and visualization."""
        return (self.q_a + self.q_b) / 2

    @q.setter
    def q(self, value: np.ndarray) -> None:
        self._q = value

    def learn(self, state, action, reward, next_state, done, next_action=None) -> None:
        del next_action
        if self.rng.random() < 0.5:
            update, evaluate = self.q_a, self.q_b
        else:
            update, evaluate = self.q_b, self.q_a
        if done:
            target = reward
        else:
            best_next = int(np.argmax(update[next_state]))
            target = reward + self.discount * float(evaluate[next_state, best_next])
        update[state, action] += self.learning_rate * (target - update[state, action])


class ExpectedSARSAAgent(_TabularAgent):
    """Expected SARSA with an epsilon-greedy expectation at the next state."""

    def learn(self, state, action, reward, next_state, done, next_action=None) -> None:
        del next_action
        target = reward
        if not done:
            values = self.q[next_state]
            best = values == values.max()
            probabilities = np.full(values.size, self.epsilon / values.size)
            probabilities[best] += (1.0 - self.epsilon) / int(best.sum())
            target += self.discount * float(np.dot(probabilities, values))
        self.q[state, action] += self.learning_rate * (target - self.q[state, action])


class SARSAAgent(_TabularAgent):
    """On-policy SARSA; pass the policy's selected ``next_action`` to learn."""

    def learn(self, state, action, reward, next_state, done, next_action=None) -> None:
        if not done and next_action is None:
            raise ValueError("SARSA requires next_action when done is False")
        target = reward if done else reward + self.discount * self.q[next_state, next_action]
        self.q[state, action] += self.learning_rate * (target - self.q[state, action])


class SarsaLambdaAgent(_TabularAgent):
    """SARSA(lambda) with replacing eligibility traces."""

    def __init__(self, *args, trace_decay: float = 0.9, **kwargs) -> None:
        if not 0 <= trace_decay <= 1:
            raise ValueError("trace_decay must be in [0, 1]")
        super().__init__(*args, **kwargs)
        self.trace_decay = trace_decay
        self.traces = np.zeros_like(self.q)

    def learn(self, state, action, reward, next_state, done, next_action=None) -> None:
        if not done and next_action is None:
            raise ValueError("SARSA(lambda) requires next_action when done is False")
        target = reward if done else reward + self.discount * self.q[next_state, next_action]
        delta = target - self.q[state, action]
        self.traces[state, action] = 1.0
        self.q += self.learning_rate * delta * self.traces
        self.traces *= self.discount * self.trace_decay
        if done:
            self.traces.fill(0.0)


class MonteCarloAgent(_TabularAgent):
    """First-visit on-policy Monte Carlo control with epsilon-greedy actions."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._episode: list[tuple[int, int, float]] = []
        self._visits = np.zeros_like(self.q, dtype=np.int64)

    def learn(self, state, action, reward, next_state, done, next_action=None) -> None:
        del next_state, next_action
        self._episode.append((int(state), int(action), float(reward)))
        if not done:
            return

        returns = [0.0] * len(self._episode)
        total = 0.0
        for index in range(len(self._episode) - 1, -1, -1):
            total = self._episode[index][2] + self.discount * total
            returns[index] = total

        visited: set[tuple[int, int]] = set()
        for (episode_state, episode_action, _), episode_return in zip(self._episode, returns):
            pair = (episode_state, episode_action)
            if pair in visited:
                continue
            visited.add(pair)
            self._visits[pair] += 1
            count = self._visits[pair]
            self.q[pair] += (episode_return - self.q[pair]) / count
        self._episode.clear()


# Short aliases for common names used in RL notes.
DoubleQAgent = DoubleQLearningAgent
ExpectedQAgent = ExpectedSARSAAgent

AGENTS: dict[str, type[_TabularAgent]] = {
    "q": QLearningAgent,
    "q_learning": QLearningAgent,
    "dq": DoubleQLearningAgent,
    "double_q": DoubleQLearningAgent,
    "eq": ExpectedSARSAAgent,
    "expected_sarsa": ExpectedSARSAAgent,
    "sarsa": SARSAAgent,
    "sarsa_lambda": SarsaLambdaAgent,
    "mc": MonteCarloAgent,
    "monte_carlo": MonteCarloAgent,
}


def make_agent(method: str, *args, **kwargs) -> _TabularAgent:
    """Build one algorithm by name, for example ``make_agent('double_q', ...)``."""
    try:
        agent_type = AGENTS[method.lower()]
    except KeyError as exc:
        choices = ", ".join(sorted(AGENTS))
        raise ValueError(f"unknown method {method!r}; choose from: {choices}") from exc
    return agent_type(*args, **kwargs)
