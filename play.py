"""Run a trained Q-learning policy in the CliffWalking window."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

import cliffwalk as gym


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "q_table",
        nargs="?",
        type=Path,
        default=Path("runs/cliffwalk/q_table.npy"),
        help="saved Q-table (default: runs/cliffwalk/q_table.npy)",
    )
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=500)
    args = parser.parse_args()
    if args.max_steps < 1:
        parser.error("--max-steps must be positive")

    if not args.q_table.is_file():
        parser.error(f"Q-table not found: {args.q_table}; train first with 'uv run train.py'")

    q_table = np.load(args.q_table)
    env = gym.make("cliffwalk", rows=args.rows, cols=args.cols, render_mode="human")
    expected_shape = (env.n_states, env.n_actions)
    if q_table.shape != expected_shape:
        env.close()
        parser.error(f"Q-table shape is {q_table.shape}; expected {expected_shape}")

    total_reward = 0.0
    terminated = truncated = False
    try:
        observation, _ = env.reset()
        for _ in range(args.max_steps):
            action = int(np.argmax(q_table[observation]))
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        else:
            truncated = True

        print(
            f"Episode finished: reward={total_reward:g}, reached_goal={terminated}, "
            f"fell={info['fell']}, truncated={truncated}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
