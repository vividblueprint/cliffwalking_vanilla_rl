"""Simple CliffWalking environment and Q-learning agent."""

from .agent import QLearningAgent
from .agent_multi import (
    DoubleQLearningAgent,
    ExpectedSARSAAgent,
    MonteCarloAgent,
    SARSAAgent,
    SarsaLambdaAgent,
    make_agent,
)
from .env import CliffWalkingEnv


def make(environment_id: str, **kwargs: object) -> CliffWalkingEnv:
    """Create a CliffWalking environment using a Gym-style factory call."""
    if environment_id not in ("cliffwalk", "CliffWalking-v0"):
        raise ValueError(f"Unknown environment ID: {environment_id}")
    return CliffWalkingEnv(**kwargs)


__all__ = [
    "CliffWalkingEnv",
    "DoubleQLearningAgent",
    "ExpectedSARSAAgent",
    "MonteCarloAgent",
    "QLearningAgent",
    "SARSAAgent",
    "SarsaLambdaAgent",
    "make",
    "make_agent",
]
