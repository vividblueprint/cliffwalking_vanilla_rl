"""Train Q-learning on the CliffWalking grid."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

from cliffwalk import CliffWalkingEnv, make_agent
from cliffwalk.agent_multi import SARSAAgent, SarsaLambdaAgent
from cliffwalk.visualize import draw_board


def train(args: argparse.Namespace):
    env = CliffWalkingEnv(args.rows, args.cols, args.render_mode)
    agent = make_agent(
        args.agent,
        env.n_states,
        env.n_actions,
        args.alpha,
        args.gamma,
        args.epsilon,
        args.epsilon_min,
        args.epsilon_decay,
        args.seed,
    )
    returns = []

    for episode in range(args.episodes):
        state, _ = env.reset(seed=args.seed if episode == 0 else None)
        total = 0.0
        action = agent.act(state)
        for step_index in range(args.max_steps):
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated or step_index == args.max_steps - 1
            next_action = None
            if not done and isinstance(agent, (SARSAAgent, SarsaLambdaAgent)):
                next_action = agent.act(next_state)
            agent.learn(
                state,
                action,
                reward,
                next_state,
                done,
                next_action=next_action,
            )
            state = next_state
            total += reward
            if done:
                break
            action = next_action if next_action is not None else agent.act(state)

        agent.decay_exploration()
        returns.append(total)
        if (episode + 1) % 200 == 0 or episode == 0:
            average = np.mean(returns[-50:])
            print(f"Episode {episode + 1}/{args.episodes}: average return {average:.1f}")

    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=True)
    np.save(output / "q_table.npy", agent.q)
    image = draw_board(env, agent.q)
    cv2.imwrite(str(output / "policy.png"), image)
    if args.show:
        cv2.imshow("CliffWalking Q-learning", image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    print(f"Saved Q-table and policy image to {output}")
    return agent, returns


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episodes", type=int, default=500)
    parser.add_argument("--rows", type=int, default=4)
    parser.add_argument("--cols", type=int, default=12)
    parser.add_argument("--max-steps", type=int, default=1000)
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--gamma", type=float, default=1.0)
    parser.add_argument("--epsilon", type=float, default=1.0)
    parser.add_argument("--epsilon-min", type=float, default=0.05)
    parser.add_argument("--epsilon-decay", type=float, default=0.995)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--render_mode", type=str, default=None)
    parser.add_argument(
        "--agent",
        choices=("q_learning", "double_q", "expected_sarsa", "sarsa", "sarsa_lambda", "monte_carlo"),
        default="q_learning",
        help="tabular control algorithm",
    )
    parser.add_argument("--out", type=Path, default=Path("runs/cliffwalk"))
    parser.add_argument("--show", action="store_true")
    args = parser.parse_args()
    if args.episodes < 1 or args.max_steps < 1:
        parser.error("episodes and max-steps must be positive")
    train(args)


if __name__ == "__main__":
    main()
