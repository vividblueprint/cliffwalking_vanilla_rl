"""Run one random-action episode using the Gym-style API."""

import cliffwalk as gym


def main() -> None:
    env = gym.make("cliffwalk", render_mode="human")
    total_reward = 0.0
    terminated = truncated = False

    try:
        observation, info = env.reset()
        for step in range(100):
            action = env.action_space.sample()
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            if terminated or truncated:
                break
        else:
            truncated = True

        print(
            f"Episode finished! Total reward: {total_reward:g}; "
            f"reached goal: {terminated}; fell: {info['fell']}"
        )
    finally:
        env.close()


if __name__ == "__main__":
    main()
