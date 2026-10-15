"""The deterministic CliffWalking grid environment."""

from __future__ import annotations

import numpy as np


class DiscreteSpace:
    """Small discrete space with Gym-style ``sample`` and ``seed``."""

    def __init__(self, n: int) -> None:
        self.n = n
        self._rng = np.random.default_rng()

    def sample(self) -> int:
        return int(self._rng.integers(self.n))

    def seed(self, seed: int | None = None) -> None:
        self._rng = np.random.default_rng(seed)


class CliffWalkingEnv:
    """Grid with a start, goal, and cliff along the bottom row."""

    metadata = {"render_modes": ["human"], "render_fps": 4}

    # Actions: up, right, down, left.
    DELTAS = ((-1, 0), (0, 1), (1, 0), (0, -1))

    def __init__(
        self, rows: int = 4, cols: int = 12, render_mode: str | None = None
    ) -> None:
        if rows < 2 or cols < 3:
            raise ValueError("rows must be >= 2 and cols must be >= 3")
        if render_mode not in (None, "human"):
            raise ValueError("render_mode must be None or 'human'")
        self.rows, self.cols = rows, cols
        self.render_mode = render_mode
        self.start = (rows - 1, 0)
        self.goal = (rows - 1, cols - 1)
        self.cliff = {(rows - 1, col) for col in range(1, cols - 1)}
        self.n_states = rows * cols
        self.n_actions = len(self.DELTAS)
        self.action_space = DiscreteSpace(self.n_actions)
        self.observation_space = DiscreteSpace(self.n_states)
        self.state = self.state_id(self.start)
        self._window_open = False

    def state_id(self, position: tuple[int, int]) -> int:
        row, col = position
        return row * self.cols + col

    def position(self, state: int) -> tuple[int, int]:
        if not 0 <= state < self.n_states:
            raise ValueError(f"state must be in [0, {self.n_states})")
        return divmod(int(state), self.cols)

    def reset(
        self, *, seed: int | None = None, options: dict | None = None
    ) -> tuple[int, dict]:
        del options
        if seed is not None:
            self.action_space.seed(seed)
            self.observation_space.seed(seed)
        self.state = self.state_id(self.start)
        if self.render_mode == "human":
            self.render()
        return self.state, {}

    def step(self, action: int) -> tuple[int, float, bool, bool, dict]:
        if not 0 <= action < self.n_actions:
            raise ValueError(f"action must be in [0, {self.n_actions})")
        row, col = self.position(self.state)
        dr, dc = self.DELTAS[action]
        next_position = (
            min(max(row + dr, 0), self.rows - 1),
            min(max(col + dc, 0), self.cols - 1),
        )

        if next_position in self.cliff:
            self.state = self.state_id(self.start)
            result = (self.state, -100.0, False, False, {"fell": True})
        else:
            self.state = self.state_id(next_position)
            terminated = next_position == self.goal
            result = (self.state, -1.0, terminated, False, {"fell": False})

        if self.render_mode == "human":
            self.render()
        return result

    def render(self) -> None:
        """Show the current board when ``render_mode='human'``."""
        if self.render_mode == "human":
            import cv2

            from .visualize import draw_board

            cv2.imshow(
                "CliffWalking environment",
                draw_board(self, agent_position=self.position(self.state)),
            )
            cv2.waitKey(max(1, 1000 // self.metadata["render_fps"]))
            self._window_open = True

    def close(self) -> None:
        """Close the rendering window, if it was opened."""
        if self._window_open:
            import cv2

            cv2.destroyWindow("CliffWalking environment")
            self._window_open = False
